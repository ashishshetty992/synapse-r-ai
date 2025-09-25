// apps/http/src/index.ts
import Fastify from 'fastify';
import cors from '@fastify/cors';
import rateLimit from '@fastify/rate-limit';
import { IQLSchema, UQLSchema } from '@nauvra/core';
import { compileIQLtoUQL } from '@nauvra/planner';
import { loadOrBuildUEM } from '@nauvra/uem';
import { MySQLAdapter } from '@nauvra/uql-mysql';
import { MongoAdapter } from '@nauvra/uql-mongo';
import { ZodError } from 'zod';
import fs from 'fs';
import path from 'path';

// helpers (put near top of file)
function shouldDebug(req: any) {
  const q = String((req.query as any)?.debug ?? '');
  const h = String(req.headers['x-debug'] ?? '');
  return q === '1' || h === '1' || process.env.DEBUG_ERRORS === '1';
}

// Map alternate entity prefixes -> canonical Phase-2 names
const ENTITY_ALIASES: Record<string, string> = {
  'orders': 'sales_order',
  // add more if needed, e.g. 'order': 'sales_order'
};

function rewriteFieldName(name: string): string {
  if (typeof name !== 'string') return name as any;
  for (const [from, to] of Object.entries(ENTITY_ALIASES)) {
    if (name === from) return to;                         // bare entity
    if (name.startsWith(from + '.')) return to + name.slice(from.length);
  }
  return name;
}

function rewriteIQLAliases(iql: any): any {
  if (!iql || typeof iql !== 'object') return iql;

  const out = JSON.parse(JSON.stringify(iql));

  // target / targets
  if (typeof out.target === 'string') out.target = rewriteFieldName(out.target);
  if (Array.isArray(out.targets)) out.targets = out.targets.map(rewriteFieldName);

  // select {expr}
  if (Array.isArray(out.select)) {
    out.select = out.select.map((s: any) =>
      typeof s?.expr === 'string' ? { ...s, expr: rewriteFieldName(s.expr) } : s
    );
  }

  // filters: recursive walker for nested and/or
  function walkFilter(f: any): any {
    if (!f || typeof f !== 'object') return f;
    if (Array.isArray(f.and)) return { ...f, and: f.and.map(walkFilter) };
    if (Array.isArray(f.or))  return { ...f, or:  f.or.map(walkFilter) };
    if (typeof f.field === 'string') return { ...f, field: rewriteFieldName(f.field) };
    return f;
  }
  if (Array.isArray(out.filters)) out.filters = out.filters.map(walkFilter);

  // orderBy {field}
  if (Array.isArray(out.orderBy)) {
    out.orderBy = out.orderBy.map((o: any) =>
      typeof o?.field === 'string' ? { ...o, field: rewriteFieldName(o.field) } : o
    );
  }

  return out;
}

function classifyError(e: unknown) {
  const msg = (e as any)?.message ?? String(e);
  let code = 'NQL_INTERNAL';
  if ((e as any)?.name === 'ZodError') code = 'VALIDATION';
  else if (/translate|compile|planner/i.test(msg)) code = 'TRANSLATION';
  else if (/sql|mongo|adapter|driver|pool|connection/i.test(msg)) code = 'ADAPTER';
  else if (/syntax|table|collection|column|index|constraint/i.test(msg)) code = 'DB';
  const status =
    code === 'VALIDATION' ? 400 :
    code === 'TRANSLATION' ? 422 :
    code === 'ADAPTER' ? 502 :
    code === 'DB' ? 503 : 500;
  return { code, status, message: msg };
}

async function main() {
  const app = Fastify({ 
    logger: true,
    bodyLimit: 1_000_000 // 1MB; adjust if needed
  });

  // Register CORS with environment-based configuration
  await app.register(cors, {
    origin: (origin, cb) => {
      const allow = (process.env.CORS_ORIGIN || '').split(',').map(s=>s.trim()).filter(Boolean);
      if (!origin || allow.length === 0 || allow.includes(origin)) return cb(null, true);
      cb(new Error('CORS not allowed'), false);
    },
    credentials: true
  });

  // Register rate limiting
  await app.register(rateLimit, {
    max: 600,
    timeWindow: '1 minute',
  });

  // ---- init UEM + adapters ----
  const uem = await loadOrBuildUEM();

  const mysql = new MySQLAdapter();
  await mysql.init(uem, {
    uri: process.env.MYSQL_URI ?? 'mysql://root:root@127.0.0.1:3306/nauvra',
  });

  const mongo = new MongoAdapter();
  await mongo.init(uem, {
    uri: process.env.MONGO_URI ?? 'mongodb://root:root@127.0.0.1:27017/?authSource=admin',
    db: process.env.MONGO_DB ?? 'nauvra',
  });

  // ---- env thread-through for time handling ----
  const ENV_TIMEZONE = process.env.TIMEZONE;
  const ENV_WEEK_START = process.env.WEEK_START; // "mon" | "sun" (adapter normalizes)

  // Add request ID to all responses
  app.addHook('onSend', async (req, reply, payload) => {
    reply.header('x-request-id', req.id);
    return payload;
  });

  app.setErrorHandler((err, req, rep) => {
    req.log.error({ err, requestId: req.id }, 'request-error');
    const { code, status, message } = classifyError(err);
    const payload: any = {
      code,
      message: 'Request failed',
      error: message,
      requestId: req.id,
    };
    if (shouldDebug(req)) payload.trace = { errorCode: code };
    rep.status(status).send(payload);
  });

  app.log.info(
    {
      mysql_uri: process.env.MYSQL_URI ? 'env:MYSQL_URI' : 'default',
      mongo_uri: process.env.MONGO_URI ? 'env:MONGO_URI' : 'default',
      mongo_db: process.env.MONGO_DB ?? 'n/a',
      timezone: ENV_TIMEZONE ?? 'unset',
      week_start: ENV_WEEK_START ?? 'unset',
    },
    'adapter-config'
  );

  // small helper to pick adapter
  function pickAdapter(h: Record<string, any>, q?: Record<string, any>) {
    const target = ((q?.target as string) ?? (h['x-target'] as string) ?? 'mysql').toLowerCase();
    if (target === 'mongo' || target === 'mongodb') return mongo;
    return mysql;
  }

  // helper to detect UQL vs IQL
  const isUQL = (body: any) =>
    body && typeof body === 'object' && body.version === 'uql/0.1' && !!body.entity;

  // small helper to read planner knobs from headers
  function readPlannerKnobs(h: Record<string, any>) {
    const modeRaw = (h['x-planner-mode'] as string) ?? (process.env.PLANNER_MODE as any) ?? 'beam';
    const plannerMode: 'deterministic'|'beam' = modeRaw === 'deterministic' ? 'deterministic' : 'beam';

    const tauNum = Number(h['x-planner-tau']);
    const tau = Number.isFinite(tauNum) ? Math.min(Math.max(tauNum, 0.05), 5) : 0.7;

    const bwNum = Number(h['x-planner-beam']);
    const beamWidth = Number.isInteger(bwNum) && bwNum > 0 && bwNum <= 16 ? bwNum : 4;

    const vec = h['x-intent-vec'] ? String(h['x-intent-vec']) : undefined;
    let intentVec: number[] | undefined;
    try {
      if (vec) {
        const parsed = JSON.parse(vec);
        const arr = Array.isArray(parsed?.vec) ? parsed.vec : parsed;
        if (Array.isArray(arr) && arr.every((x) => typeof x === 'number')) intentVec = arr;
      }
    } catch {/* ignore */}
    return { plannerMode, tau, beamWidth, intentVec };
  }

  // Merge timezone/weekStart into uql.hints (additive, do not overwrite existing)
  function injectTimeHints(uql: any): any {
    const tz = ENV_TIMEZONE && typeof ENV_TIMEZONE === 'string' ? ENV_TIMEZONE : undefined;
    const wkRaw = (ENV_WEEK_START || '').toLowerCase();
    const wk = wkRaw.startsWith('sun') ? 'sun' : wkRaw.startsWith('mon') ? 'mon' : undefined;
    if (!tz && !wk) return uql;
    return { ...uql, hints: { ...(uql.hints ?? {}), ...(tz ? { timezone: tz } : {}), ...(wk ? { weekStart: wk } : {}) } };
  }

  // Examples caching
  const EXAMPLES_DIR = path.resolve(new URL('../../../examples/iql', import.meta.url).pathname);
  const examplesCache = new Map<string, any>();

  // ------------------------------------
  // GET /examples  (serve examples/iql/*.json)
  // ------------------------------------
  app.get('/examples', async (_req, reply) => {
    if (!fs.existsSync(EXAMPLES_DIR)) return reply.send({ files: [], note: 'examples/iql not found' });
    const files = fs.readdirSync(EXAMPLES_DIR).filter(f => f.endsWith('.json')).sort();
    const items = files.map((f) => {
      try {
        if (!examplesCache.has(f)) {
          const text = fs.readFileSync(path.join(EXAMPLES_DIR, f), 'utf8');
          examplesCache.set(f, JSON.parse(text));
        }
        return { name: f, json: examplesCache.get(f) };
      } catch (e: any) {
        return { name: f, error: e?.message ?? String(e) };
      }
    });
    return reply.send({ files: items });
  });

  // ------------------------------------
  // GET /examples/:name  (fetch single example)
  // ------------------------------------
  app.get('/examples/:name', async (req, reply) => {
    const raw = String((req.params as any).name);
    const refresh = String((req.query as any)?.refresh ?? '') === '1';
    
    if (!/^[a-z0-9._-]+\.json$/i.test(raw)) {
      return reply.code(400).send({ code: 'EXAMPLES_ERROR', message: 'invalid filename' });
    }
    const file = path.join(EXAMPLES_DIR, raw);
    const abs = path.resolve(file);
    if (!abs.startsWith(EXAMPLES_DIR)) return reply.code(400).send({ code: 'EXAMPLES_ERROR', message: 'invalid path' });
    if (!fs.existsSync(abs)) return reply.code(404).send({ code: 'NOT_FOUND' });

    try {
      if (refresh) examplesCache.delete(raw);
      if (!examplesCache.has(raw)) {
        examplesCache.set(raw, JSON.parse(fs.readFileSync(abs, 'utf8')));
      }
      return reply.send(examplesCache.get(raw));
    } catch (e: any) {
      return reply.code(500).send({ code: 'EXAMPLES_ERROR', message: e?.message ?? String(e) });
    }
  });

  // ------------------------------------
  // POST /query  (UQL or IQL -> execute)
  // ------------------------------------
  app.post('/query', {
    schema: {
      querystring: {
        type: 'object',
        properties: { debug: { type: 'string' }, target: { type: 'string' } },
        additionalProperties: true
      },
      headers: {
        type: 'object',
        properties: {
          'x-target': { type: 'string' },
          'x-planner-mode': { type: 'string', enum: ['beam','deterministic'] },
          'x-planner-tau': { type: 'string' },
          'x-planner-beam': { type: 'string' },
          'x-intent-vec': { type: 'string' },
        },
        additionalProperties: true
      }
    }
  }, async (req, reply) => {
    const debug = shouldDebug(req);
    const t0 = Date.now();

    try {
      const adapter = pickAdapter(req.headers, req.query as any);
      const adapterName = adapter.name;
      reply.header('x-adapter', adapterName);
      const body = req.body as any;

      // Direct UQL
      if (isUQL(body)) {
        // validate UQL strictly before executing
        let uqlBody;
        try {
          uqlBody = UQLSchema.parse(body);
        } catch (e) {
          if (e instanceof ZodError) {
            const details = e.issues.map(i => ({
              path: i.path.join('.'),
              msg: i.message,
              code: i.code
            }));
            return reply.status(400).send({ code: 'VALIDATION', details });
          }
          throw e;
        }
        const uql = injectTimeHints(uqlBody);
        const planMs = Date.now() - t0;
        const t1 = Date.now();
        const result = await adapter.read(uql);
        const adapterMs = Date.now() - t1;
        const rowCount = Array.isArray(result.rows) ? result.rows.length : 0;

        const traceOut = debug
          ? { planMs, adapterMs, rowCount }
          : undefined;

        return reply.send({
          uql,
          ...(debug ? { trace: traceOut } : {}),
          rows: result.rows,
          meta: { ...result.meta, planMs, adapterMs, rowCount, adapter: adapterName }
        });
      }

      // IQL path with Zod validation -> 400
      let iql;
      try {
        iql = IQLSchema.parse(body);
      } catch (e) {
        if (e instanceof ZodError) {
          const details = e.issues.map(i => ({ path: i.path.join('.'), msg: i.message }));
          return reply.status(400).send({ code: 'VALIDATION', details });
        }
        throw e;
      }

      iql = rewriteIQLAliases(iql);

      const { plannerMode, tau, beamWidth, intentVec } = readPlannerKnobs(req.headers);
      const { uql, trace } = compileIQLtoUQL(iql, uem, { plannerMode, tau, beamWidth, intentVec });
      const uqlWithHints = injectTimeHints(uql);

      const planMs = Date.now() - t0;
      const t1 = Date.now();
      const result = await adapter.read(uqlWithHints as any);
      const adapterMs = Date.now() - t1;
      const rowCount = Array.isArray(result.rows) ? result.rows.length : 0;

      const traceOut = debug
        ? { ...(trace ?? {}), planMs, adapterMs, rowCount }
        : undefined;

      return reply.send({
        uql: uqlWithHints,
        ...(debug ? { trace: traceOut } : {}),
        rows: result.rows,
        meta: { ...result.meta, planMs, adapterMs, rowCount, adapter: adapterName }
      });

    } catch (e: any) {
      const { code, status, message } = classifyError(e);
      const dbg = shouldDebug(req);
      return reply.code(status).send({
        code,
        message: 'Request failed',
        error: message,
        requestId: (req as any).id,
        ...(dbg ? { trace: { errorCode: code } } : {})
      });
    }
  });

  // --------------------------------------------------
  // POST /compile  (IQL -> UQL; don't execute DB call)
  // --------------------------------------------------
  app.post('/compile', {
    schema: {
      querystring: {
        type: 'object',
        properties: { debug: { type: 'string' } },
        additionalProperties: true
      },
      headers: {
        type: 'object',
        properties: {
          'x-planner-mode': { type: 'string', enum: ['beam','deterministic'] },
          'x-planner-tau': { type: 'string' },
          'x-planner-beam': { type: 'string' },
          'x-intent-vec': { type: 'string' },
        },
        additionalProperties: true
      }
    }
  }, async (req, reply) => {
    try {
      let iql;
      try {
        iql = IQLSchema.parse(req.body);
      } catch (e) {
        if (e instanceof ZodError) {
          const details = e.issues.map(i => ({ path: i.path.join('.'), msg: i.message }));
          return reply.status(400).send({ code: 'VALIDATION', details });
        }
        throw e;
      }

      iql = rewriteIQLAliases(iql);

      const { plannerMode, tau, beamWidth, intentVec } = readPlannerKnobs(req.headers);
      const t0 = Date.now();
      const { uql, trace } = compileIQLtoUQL(iql, uem, { plannerMode, tau, beamWidth, intentVec });
      const planMs = Date.now() - t0;

      return reply.send({ 
        uql: injectTimeHints(uql), 
        ...(shouldDebug(req) ? { trace } : {}), 
        meta: { planMs, adapter: null } 
      });
    } catch (e: any) {
      const { code, status, message } = classifyError(e);
      const dbg = shouldDebug(req);
      return reply.code(status).send({
        code,
        message: 'Request failed',
        error: message,
        requestId: (req as any).id,
        ...(dbg ? { trace: { errorCode: code } } : {})
      });
    }
  });

  // -------------------------------------------------------------
  // POST /explain  (UQL or IQL -> adapter.explain with joins etc.)
  // -------------------------------------------------------------
  app.post('/explain', {
    schema: {
      querystring: {
        type: 'object',
        properties: { debug: { type: 'string' }, target: { type: 'string' } },
        additionalProperties: true
      },
      headers: {
        type: 'object',
        properties: {
          'x-target': { type: 'string' },
          'x-planner-mode': { type: 'string', enum: ['beam','deterministic'] },
          'x-planner-tau': { type: 'string' },
          'x-planner-beam': { type: 'string' },
          'x-intent-vec': { type: 'string' },
        },
        additionalProperties: true
      }
    }
  }, async (req, reply) => {
    const debug = shouldDebug(req);
    const t0 = Date.now();

    try {
      const adapter = pickAdapter(req.headers, req.query as any);
      const adapterName = adapter.name;
      reply.header('x-adapter', adapterName);
      const body = req.body as any;

      if (isUQL(body)) {
        // validate UQL strictly before executing
        let uqlBody;
        try {
          uqlBody = UQLSchema.parse(body);
        } catch (e) {
          if (e instanceof ZodError) {
            const details = e.issues.map(i => ({
              path: i.path.join('.'),
              msg: i.message,
              code: i.code
            }));
            return reply.status(400).send({ code: 'VALIDATION', details });
          }
          throw e;
        }
        const uql = injectTimeHints(uqlBody);
        const planMs = Date.now() - t0;
        const t1 = Date.now();
        const adapterExplain = await adapter.explain(uql);
        const adapterMs = Date.now() - t1;

        const traceOut = debug
          ? { planMs, adapterMs }
          : undefined;

        return reply.send({
          uql,
          ...(debug ? { trace: traceOut } : {}),
          adapterExplain,
          meta: { planMs, adapterMs, adapter: adapterName }
        });
      }

      let iql;
      try {
        iql = IQLSchema.parse(body);
      } catch (e) {
        if (e instanceof ZodError) {
          const details = e.issues.map(i => ({ path: i.path.join('.'), msg: i.message }));
          return reply.status(400).send({ code: 'VALIDATION', details });
        }
        throw e;
      }

      iql = rewriteIQLAliases(iql);

      const { plannerMode, tau, beamWidth, intentVec } = readPlannerKnobs(req.headers);
      const { uql, trace } = compileIQLtoUQL(iql, uem, { plannerMode, tau, beamWidth, intentVec });
      const uqlWithHints = injectTimeHints(uql);
      const planMs = Date.now() - t0;

      const t1 = Date.now();
      const adapterExplain = await adapter.explain(uqlWithHints as any);
      const adapterMs = Date.now() - t1;

      const traceOut = debug
        ? { ...(trace ?? {}), planMs, adapterMs }
        : undefined;

      return reply.send({
        uql: uqlWithHints,
        ...(debug ? { trace: traceOut } : {}),
        adapterExplain,
        meta: { planMs, adapterMs, adapter: adapterName }
      });

    } catch (e: any) {
      const { code, status, message } = classifyError(e);
      const dbg = shouldDebug(req);
      return reply.code(status).send({
        code,
        message: 'Request failed',
        error: message,
        requestId: (req as any).id,
        ...(dbg ? { trace: { errorCode: code } } : {})
      });
    }
  });

  // ------------------------------------------------------------------
  // POST /seed  (create demo data in MySQL & Mongo with basic indexes)
  // ------------------------------------------------------------------
  app.register(async function (instance) {
    instance.route({
      method: 'POST', 
      url: '/seed',
      config: { rateLimit: { max: 5, timeWindow: '1 minute' } },
      schema: {
        body: {
          type: 'object',
          properties: { backend: { type: 'string', enum: ['mysql', 'mongo', 'both'] } },
          additionalProperties: true
        }
      },
      handler: async (req, reply) => {
    if (process.env.NODE_ENV === 'production' && process.env.ALLOW_SEED !== 'true') {
      return reply.code(403).send({ code: 'FORBIDDEN', message: 'seeding disabled in production' });
    }

    const { backend = 'both' } = (req.body as any) ?? {};
    const did: Record<string, number> = {};

    // ---- tiny customer master ----
    const customers = [
      { id: 101, name: 'Acme Corp',      segment: 'Enterprise' },
      { id: 102, name: 'Globex',         segment: 'Enterprise' },
      { id: 103, name: 'Soylent Co',     segment: 'Mid-Market' },
      { id: 104, name: 'Initech',        segment: 'Mid-Market' },
      { id: 105, name: 'Umbrella LLC',   segment: 'SMB' },
      { id: 106, name: 'Hooli',          segment: 'Enterprise' },
      { id: 107, name: 'Vehement',       segment: 'SMB' },
    ];

    // ---- orders (referencing the customer ids above) ----
    const sample = [
      { id: 1, customer_id: 101, total_amount: 199.99, created_at: '2025-08-05 10:00:00', country: 'IN', shipping_city: 'Bengaluru' },
      { id: 2, customer_id: 102, total_amount: 299.99, created_at: '2025-08-12 12:00:00', country: 'IN', shipping_city: 'Mumbai' },
      { id: 3, customer_id: 103, total_amount: 149.99, created_at: '2025-08-20 09:30:00', country: 'IN', shipping_city: 'Bengaluru' },
      { id: 4, customer_id: 104, total_amount: 199.99, created_at: '2025-08-22 14:30:00', country: 'IN', shipping_city: 'Delhi' },
      { id: 5, customer_id: 105, total_amount: 219.00, created_at: '2025-08-25 18:15:00', country: 'IN', shipping_city: 'Delhi' },
      { id: 6, customer_id: 106, total_amount: 179.25, created_at: '2025-08-26 08:45:00', country: 'IN', shipping_city: 'Pune' },
      { id: 7, customer_id: 107, total_amount: 399.00, created_at: '2025-08-27 13:10:00', country: 'IN', shipping_city: 'Chennai' }
    ];

    // ---------- MySQL ----------
    if (backend === 'mysql' || backend === 'both') {
      const pool = (mysql as any).pool;

      await pool.query(`CREATE TABLE IF NOT EXISTS customer (
        id INT PRIMARY KEY,
        name VARCHAR(100),
        segment VARCHAR(50),
        KEY idx_customer_id (id),
        KEY idx_customer_name (name)
      ) ENGINE=InnoDB`);

      await pool.query(`CREATE TABLE IF NOT EXISTS sales_order (
        id INT PRIMARY KEY,
        customer_id INT NOT NULL,
        total_amount DECIMAL(10,2),
        created_at DATETIME NOT NULL,
        country VARCHAR(2),
        shipping_city VARCHAR(128),
        KEY idx_created_at (created_at),
        KEY idx_country (country),
        KEY idx_city (shipping_city),
        KEY idx_customer_id (customer_id)
      ) ENGINE=InnoDB`);

      await pool.query(`SET FOREIGN_KEY_CHECKS = 0`);
      await pool.query(`TRUNCATE TABLE sales_order`);
      await pool.query(`TRUNCATE TABLE customer`);
      await pool.query(`SET FOREIGN_KEY_CHECKS = 1`);

      await pool.query(
        `INSERT INTO customer (id, name, segment) VALUES ?`,
        [customers.map(c => [c.id, c.name, c.segment])]
      );

      await pool.query(
        `INSERT INTO sales_order (id, customer_id, total_amount, created_at, country, shipping_city) VALUES ?`,
        [sample.map(r => [r.id, r.customer_id, r.total_amount, r.created_at, r.country, r.shipping_city])]
      );

      try {
        await pool.query(
          `ALTER TABLE sales_order
           ADD CONSTRAINT fk_sales_order_customer
           FOREIGN KEY (customer_id) REFERENCES customer(id)`
        );
      } catch {}

      did.mysql = sample.length;
    }

    // ---------- Mongo ----------
    if (backend === 'mongo' || backend === 'both') {
      const db = (mongo as any).cli.db((mongo as any).dbName ?? 'nauvra');

      const coll = db.collection('sales_order');
      await coll.deleteMany({});

      const cust = db.collection('customer');
      await cust.deleteMany({});

      await cust.createIndex({ id: 1 }, { unique: true });
      await cust.createIndex({ name: 1 });
      await cust.insertMany(customers);

      await coll.createIndex({ created_at: 1 });
      await coll.createIndex({ country: 1 });
      await coll.createIndex({ shipping_city: 1 });
      await coll.createIndex({ customer_id: 1 });

      await coll.insertMany(sample.map(r => ({
        ...r,
        created_at: new Date(r.created_at + 'Z')
      })));

      did.mongo = sample.length;
    }

    return reply.send({ seeded: did });
      }
    });
  });

  // ---------------------------------------------------------------------------------
  // POST /parity  (run same IQL on both backends; compare rows with ordering applied)
  // ---------------------------------------------------------------------------------
  app.post('/parity', {
    schema: {
      headers: {
        type: 'object',
        properties: {
          'x-planner-mode': { type: 'string', enum: ['beam','deterministic'] },
          'x-planner-tau': { type: 'string' },
          'x-planner-beam': { type: 'string' },
          'x-intent-vec': { type: 'string' },
        },
        additionalProperties: true
      }
    }
  }, async (req, reply) => {
    try {
      const body = (req.body ?? {}) as any;
      const iql = 'iql' in body ? body.iql : body;

      let parsed;
      try {
        parsed = IQLSchema.parse(iql);
      } catch (e) {
        if (e instanceof ZodError) {
          const details = e.issues.map(i => ({ path: i.path.join('.'), msg: i.message }));
          return reply.status(400).send({ code: 'VALIDATION', details });
        }
        throw e;
      }

      const { plannerMode, tau, beamWidth, intentVec } = readPlannerKnobs(req.headers);
      const { uql, trace } = compileIQLtoUQL(parsed, uem, { plannerMode, tau, beamWidth, intentVec });
      const uqlH = injectTimeHints(uql);

      const [mysqlRes, mongoRes] = await Promise.all([ 
        mysql.read(uqlH as any), 
        mongo.read(uqlH as any) 
      ]);

      const normalizeCell = (v: any): any => {
        if (v instanceof Date) return v.toISOString();
        if (typeof v === 'number' && Number.isFinite(v)) return Number(v);
        if (v && typeof v === 'object') return JSON.parse(JSON.stringify(v)); // flattens ObjectId/Decimal128
        return v;
      };
      const norm = (rows: any[]) =>
        rows.map((r) => {
          const o: any = {};
          for (const k of Object.keys(r).sort()) o[k] = normalizeCell(r[k]);
          return o;
        });

      const L = norm(mysqlRes.rows);
      const R = norm(mongoRes.rows);
      const sameLen = L.length === R.length;
      const sameElems = sameLen && L.every((row, i) => JSON.stringify(row) === JSON.stringify(R[i]));
      const equal = sameLen && sameElems;

      let diff: any = null;
      if (!equal) {
        const firstIdx = Math.min(L.length, R.length);
        let firstMismatch = -1;
        for (let i = 0; i < firstIdx; i++) {
          if (JSON.stringify(L[i]) !== JSON.stringify(R[i])) {
            firstMismatch = i;
            break;
          }
        }
        diff = {
          equal,
          left_count: L.length,
          right_count: R.length,
          first_mismatch_index: firstMismatch,
          left_sample: L.slice(0, 10),
          right_sample: R.slice(0, 10),
        };
      }

      return reply.send({
        uql: uqlH,
        trace,
        parity: equal ? { equal: true } : diff,
        mysql: { rows: mysqlRes.rows, meta: { ...mysqlRes.meta, adapter: 'mysql' } },
        mongo: { rows: mongoRes.rows, meta: { ...mongoRes.meta, adapter: 'mongodb' } },
      });
    } catch (e: any) {
      const { code, status, message } = classifyError(e);
      const dbg = shouldDebug(req);
      return reply.code(status).send({
        code,
        message: 'Request failed',
        error: message,
        requestId: (req as any).id,
        ...(dbg ? { trace: { errorCode: code } } : {})
      });
    }
  });

  app.get('/healthz', async () => ({ ok: true }));

  app.get('/readyz', async () => {
    const [mh, gh] = await Promise.allSettled([
      (mysql as any).health?.() ?? Promise.resolve({ ok: true }),
      (mongo as any).health?.() ?? Promise.resolve({ ok: true }),
    ]);
    const ok = mh.status === 'fulfilled' && mh.value.ok && gh.status === 'fulfilled' && gh.value.ok;
    return { 
      ok, 
      mysql: mh.status === 'fulfilled' ? mh.value : { ok: false }, 
      mongo: gh.status === 'fulfilled' ? gh.value : { ok: false } 
    };
  });

  const onShutdown = async (signal: string) => {
    app.log.info({ signal }, 'shutting-down');
    try {
      await Promise.allSettled([
        (mysql as any)?.pool?.end?.(),
        (mongo as any)?.cli?.close?.(),
        app.close()
      ]);
    } finally {
      process.exit(0);
    }
  };
  process.on('SIGINT', () => onShutdown('SIGINT'));
  process.on('SIGTERM', () => onShutdown('SIGTERM'));

  await app.listen({ port: 4000, host: '0.0.0.0' });
  app.log.info('HTTP on :4000');
}

main().catch((err) => {
  // eslint-disable-next-line no-console
  console.error('Fatal boot error', err);
  process.exit(1);
});