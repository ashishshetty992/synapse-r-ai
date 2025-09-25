// packages/uql-mongodb/src/index.ts
import { MongoClient } from 'mongodb';
import type { Adapter, ReadQuery, UEM, JoinSpec } from '@nauvra/core';
import { requireIndexed } from '@nauvra/core';

type MatchExpr = Record<string, any>;

export class MongoAdapter implements Adapter {
  name: 'mongodb' = 'mongodb';
  public cli!: MongoClient;
  public dbName = 'nauvra';
  private uem!: UEM;

  async init(uem: UEM, cfg: { uri?: string; db?: string } = {}) {
    this.uem = uem;
    this.dbName = cfg.db ?? 'nauvra';
    const uri = cfg.uri ?? 'mongodb://root:root@127.0.0.1:27017/?authSource=admin';
    this.cli = await MongoClient.connect(uri);
  }

  // ---------- helpers ----------
  private isCountStar(expr: string): boolean {
    return /^count\(\*\)$/i.test(expr);
  }

  private isCountDistinct(expr: string): string | null {
    const m = expr.match(/^count_distinct\(([^)]+)\)$/i);
    return m ? m[1] : null;
  }

  private isAgg(expr: string): boolean {
    return /^(sum|avg|min|max)\(/i.test(expr) || this.isCountStar(expr) || !!this.isCountDistinct(expr);
  }

  // Map a UQL field (possibly qualified like "customer.name") to a Mongo path with $ prefix.
  // aliasMap: entityName -> lookup alias (e.g., "__j1"), baseEntity is q.entity
  private toFieldPath(field: string, aliasMap: Record<string, string>, baseEntity: string): string {
    if (!field) return field;
    if (field.includes('.')) {
      const [prefix, ...rest] = field.split('.');
      const rhs = rest.join('.');
      if (prefix === baseEntity) return `$${rhs}`; // explicit base qualifier
      const alias = aliasMap[prefix];
      if (alias) return `$${alias}.${rhs}`;
      // unknown qualifier—fall back to raw (treat as base)
      return `$${field}`;
    }
    // unqualified => base
    return `$${field}`;
  }

  // Normalize weekStart to Mongo's expected strings
  private normWeekStart(v?: string | null): 'monday'|'sunday'|undefined {
    const s = (v || '').toString().toLowerCase();
    if (s.startsWith('mon')) return 'monday';
    if (s.startsWith('sun')) return 'sunday';
    return undefined;
  }

  // Coerce date literals for consistent date handling
  private coerceDateLiteral(v: any): any {
    if (v instanceof Date) return v;
    if (typeof v === 'string') {
      if (/^\d{4}-\d{2}-\d{2}T/.test(v)) return new Date(v);                // ISO with time
      if (/^\d{4}-\d{2}-\d{2}$/.test(v))   return new Date(v + 'T00:00:00.000Z'); // date-only
    }
    return v;
  }

  // Build period switch expression for compare periods
  private buildPeriodSwitch(opts: { comparePeriods?: Array<{label:string,start:string,end:string}>,
                                   baseEntity: string,
                                   aliasMap: Record<string,string> }) {
    const cps = opts.comparePeriods || [];
    if (!cps.length) return null;

    const path = this.toFieldPath('created_at', opts.aliasMap, opts.baseEntity); // "$created_at"
    const branches = cps.map(p => ({
      case: { $and: [ { $gte: [ path, new Date(p.start) ] }, { $lt: [ path, new Date(p.end) ] } ] },
      then: p.label
    }));
    return { $switch: { branches, default: 'Other' } };
  }

  // Split match predicates into base-only vs join-referencing for optimization
  private splitMatchPredicates(match: MatchExpr): { baseOnly: MatchExpr; joinPart: MatchExpr } {
    const baseOnly: MatchExpr = {};
    const joinPart: MatchExpr = {};

    const isJoinKey = (k: string) => k.startsWith('__j') && k.includes('.');

    const walk = (src: any, dstBase: any, dstJoin: any) => {
      if (!src || typeof src !== 'object' || Array.isArray(src)) return; // arrays handled under $and/$or below

      for (const [key, val] of Object.entries(src)) {
        if (key === '$and' || key === '$or') {
          const baseArr: any[] = [];
          const joinArr: any[] = [];
          for (const item of (val as any[])) {
            const itemBase: any = {};
            const itemJoin: any = {};
            // recurse into objects; if array shows up here, we'll only assign if children create content
            if (item && typeof item === 'object') {
              walk(item, itemBase, itemJoin);
            }
            if (Object.keys(itemBase).length) baseArr.push(itemBase);
            if (Object.keys(itemJoin).length) joinArr.push(itemJoin);
          }
          if (baseArr.length) dstBase[key] = baseArr;
          if (joinArr.length) dstJoin[key] = joinArr;
        } else if (isJoinKey(key)) {
          dstJoin[key] = val;
        } else {
          dstBase[key] = val;
        }
      }
    };

    walk(match, baseOnly, joinPart);
    return { baseOnly, joinPart };
  }

  // Parse supported expr → Mongo projection/aggregation snippets
  private exprToMongo(
    expr: string,
    aliasMap: Record<string, string>,
    baseEntity: string,
    opts?: { timezone?: string; weekStart?: string }
  ): any {
    if (this.isCountStar(expr)) return null;

    const distinctField = this.isCountDistinct(expr);
    if (distinctField) {
      return { $addToSet: this.toFieldPath(distinctField, aliasMap, baseEntity) };
    }

    // date_trunc_<unit>(field)
    const m = expr.match(/^date_trunc_(day|week|month)\(([^)]+)\)$/i);
    if (m) {
      const unit = m[1].toLowerCase();
      const arg = m[2];
      const datePath = this.toFieldPath(arg, aliasMap, baseEntity); // "$t1.created_at" or "$created_at"

      const timezone = opts?.timezone || process.env.TIMEZONE || undefined;
      const startOfWeek = this.normWeekStart(opts?.weekStart || process.env.WEEK_START || undefined);

      const trunc: any = { date: datePath, unit };
      if (timezone) trunc.timezone = timezone;
      if (unit === 'week' && startOfWeek) trunc.startOfWeek = startOfWeek;

      return { $dateTrunc: trunc };
    }

    // raw field or qualified field
    if (/^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*$/.test(expr)) {
      return this.toFieldPath(expr, aliasMap, baseEntity);
    }

    // fallback: treat as "$<expr>" (best effort)
    return `$${expr}`;
  }

  private buildMatch(where: any, aliasMap: Record<string, string>, baseEntity: string): MatchExpr {
    const walk = (w: any): any => {
      if (!w) return {};
      if ('and' in w) return { $and: (w.and ?? []).map(walk) };
      if ('or' in w) return { $or: (w.or ?? []).map(walk) };

      const fieldPath = this.toFieldPath(w.field, aliasMap, baseEntity).slice(1); // remove leading '$'

      switch (w.op) {
        case 'eq': return { [fieldPath]: w.value };
        case 'neq': return { [fieldPath]: { $ne: w.value } };

        case 'between': {
          const [start, end] = w.value ?? [];
          const c = this.coerceDateLiteral.bind(this);
          return { [fieldPath]: { $gte: c(start), $lt: c(end) } };
        }

        case 'exists':
          return w.value ? { [fieldPath]: { $ne: null } } : { [fieldPath]: { $exists: false } };

        case 'gt': {
          const c = this.coerceDateLiteral.bind(this);
          return { [fieldPath]: { $gt: c(w.value) } };
        }
        case 'gte': {
          const c = this.coerceDateLiteral.bind(this);
          return { [fieldPath]: { $gte: c(w.value) } };
        }
        case 'lt': {
          const c = this.coerceDateLiteral.bind(this);
          return { [fieldPath]: { $lt: c(w.value) } };
        }
        case 'lte': {
          const c = this.coerceDateLiteral.bind(this);
          return { [fieldPath]: { $lte: c(w.value) } };
        }

        case 'in':  return { [fieldPath]: { $in:  Array.isArray(w.value) ? w.value : [w.value] } };
        case 'nin': return { [fieldPath]: { $nin: Array.isArray(w.value) ? w.value : [w.value] } };

        case 'like': {
          const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
          const raw = String(w.value ?? '');
          return { [fieldPath]: { $regex: `^${esc(raw)}` } };
        }
        case 'contains': {
          const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
          const raw = String(w.value ?? '');
          return { [fieldPath]: { $regex: esc(raw), $options: 'i' } };
        }
        case 'ilike': {
          const esc = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
          const raw = String(w.value ?? '');
          return { [fieldPath]: { $regex: `^${esc(raw)}`, $options: 'i' } };
        }

        case 'is': {
          const v = String(w.value);
          if (v === 'null')     return { [fieldPath]: null };
          if (v === 'not_null') return { [fieldPath]: { $ne: null } };
          if (v === 'true')     return { [fieldPath]: true };
          if (v === 'false')    return { [fieldPath]: false };
          throw new Error(`Unsupported is-operand for Mongo: ${w.value}`);
        }
        default:
          throw new Error(`Unsupported op for Mongo: ${w.op}`);
      }
    };
    if (!where) return {};
    const node = walk(where);
    return node && Object.keys(node).length ? node : {};
  }

  // Build $lookups/$unwinds for 1-hop joins.
  // Returns an alias map: entityName -> alias used in pipeline.
  private buildJoins(joins: JoinSpec[] | undefined): { stages: any[]; aliasMap: Record<string, string> } {
    const stages: any[] = [];
    const aliasMap: Record<string, string> = {};

    if (!joins || !joins.length) return { stages, aliasMap };

    joins.forEach((j, i) => {
      const as = `__j${i + 1}`;
      aliasMap[j.to] = as;

      stages.push({
        $lookup: {
          from: j.to,
          localField: j.on.left,    // base field in current doc
          foreignField: j.on.right, // pk in joined collection
          as
        }
      });

      stages.push({
        $unwind: {
          path: `$${as}`,
          preserveNullAndEmptyArrays: j.type === 'left'
        }
      });
    });

    return { stages, aliasMap };
  }

  async read(q: ReadQuery) {
    if (q.hints?.requireIndexedFilters) requireIndexed(this.uem, q.entity, q.where);

    const db = this.cli.db(this.dbName);
    const coll = db.collection(q.entity);
    const pipeline: any[] = [];

    // JOINs
    const { stages: joinStages, aliasMap } = this.buildJoins(q.joins);
    
    // WHERE - optimize by splitting base vs join predicates
    const match = this.buildMatch(q.where, aliasMap, q.entity);
    if (Object.keys(match).length) {
      const { baseOnly, joinPart } = this.splitMatchPredicates(match);
      
      // Push base-only filters before joins for better performance
      if (Object.keys(baseOnly).length) {
        pipeline.push({ $match: baseOnly });
      }
      
      // Add joins
      if (joinStages.length) pipeline.push(...joinStages);
      
      // Add join-referencing filters after joins
      if (Object.keys(joinPart).length) {
        pipeline.push({ $match: joinPart });
      }
    } else {
      // No filters - just add joins
      if (joinStages.length) pipeline.push(...joinStages);
    }

    // timezone + weekStart options for dateTrunc, from hints or env
    const tzOpt = (q.hints as any)?.timezone || process.env.TIMEZONE;
    const wkOpt = (q.hints as any)?.weekStart || process.env.WEEK_START;

    // Compute period switch expression once for reuse
    const periodExpr = this.buildPeriodSwitch({
      comparePeriods: (q.hints as any)?.comparePeriods,
      baseEntity: q.entity,
      aliasMap
    });

    // GROUP BY
    if (q.groupBy?.length) {
      // Build a quick map: alias -> expr
      const aliasToExpr = new Map<string, string>();
      for (const s of q.select) if (s.as) aliasToExpr.set(s.as, s.expr);
      const findSelectFor = (g: string) =>
        q.select.find(s => (s.as === g) || (s.expr === g));

      const safeKeyFor = (g: string): string => {
        const sel = findSelectFor(g);
        if (sel?.as) return sel.as;
        return g.replace(/\./g, '__');
      };

      const group: any = { };
      const isSingle = q.groupBy.length === 1;

      if (isSingle) {
        const g = q.groupBy[0];
        const sel = findSelectFor(g);
        const expr = (g === '__period__' && periodExpr)
          ? periodExpr
          : this.exprToMongo(sel ? sel.expr : g, aliasMap, q.entity, { timezone: tzOpt, weekStart: wkOpt });
        group._id = expr;
      } else {
        const id: Record<string, any> = {};
        for (const g of q.groupBy) {
          const sel = findSelectFor(g);
          const key = safeKeyFor(g);
          id[key] = (g === '__period__' && periodExpr)
            ? periodExpr
            : this.exprToMongo(sel ? sel.expr : g, aliasMap, q.entity, { timezone: tzOpt, weekStart: wkOpt });
        }
        group._id = id;
      }

      // Aggregations (count(*) / count_distinct / sum/avg/min/max)
      for (const s of q.select) {
        const alias = s.as ?? s.expr;

        if (this.isCountStar(s.expr)) {
          group[alias] = { $sum: 1 };
          continue;
        }

        const cd = this.isCountDistinct(s.expr);
        if (cd) {
          group[alias] = { $addToSet: this.toFieldPath(cd, aliasMap, q.entity) };
          continue;
        }

        // sum/avg/min/max(<inner>)
        const agg = s.expr.match(/^(sum|avg|min|max)\(([^)]+)\)$/i);
        if (agg) {
          const op = agg[1].toLowerCase();
          const inner = agg[2];
          const innerExpr = this.exprToMongo(inner, aliasMap, q.entity, { timezone: tzOpt, weekStart: wkOpt });
          const opMap: Record<string, any> = { sum: '$sum', avg: '$avg', min: '$min', max: '$max' };
          group[alias] = { [opMap[op]]: innerExpr };
        }
      }
      pipeline.push({ $group: group });

      // PROJECT
      const project: Record<string, any> = { _id: 0 };
      for (const s of q.select) {
        const alias = s.as ?? s.expr;

        if (this.isCountStar(s.expr)) {
          project[alias] = 1; // already computed in $group
          continue;
        }

        if (this.isAgg(s.expr)) {
          // count_distinct sets are arrays -> size them
          if (this.isCountDistinct(s.expr)) {
            project[alias] = { $size: `$${alias}` };
          } else {
            project[alias] = 1;
          }
          continue;
        }

        // Handle period_label(created_at) expressions
        if (/^period_label\s*\(\s*created_at\s*\)$/i.test(s.expr) && periodExpr) {
          project[alias] = periodExpr;
          continue;
        }

        // grouped key projection
        if (isSingle) {
          project[alias] = '$_id';
        } else {
          const key = (s.as ?? s.expr).replace(/\./g, '__');
          project[alias] = `$_id.${key}`;
        }
      }
      pipeline.push({ $project: project });
    } else {
      // no groupBy: simple projection
      const project: Record<string, any> = { _id: 0 };
      for (const s of q.select) {
        const alias = s.as ?? s.expr;
        const v = this.exprToMongo(s.expr, aliasMap, q.entity, { timezone: tzOpt, weekStart: wkOpt });
        if (v !== null) {
          if (/^period_label\s*\(\s*created_at\s*\)$/i.test(s.expr) && periodExpr) {
            project[alias] = periodExpr;
          } else {
            project[alias] = v;
          }
        }
      }
      pipeline.push({ $project: project });
    }

    // ORDER BY
    if (q.orderBy?.length) {
      const sort: Record<string, 1 | -1> = {};
      const selectAliases = new Set(q.select.map(s => s.as ?? s.expr));
      for (const ob of q.orderBy) {
        const key = selectAliases.has(ob.field)
          ? ob.field
          : this.toFieldPath(ob.field, aliasMap, q.entity).slice(1);
        
        if (key) sort[key] = ob.dir === 'asc' ? 1 : -1;
      }
      pipeline.push({ $sort: sort });
    }

    // OFFSET: add skip stage if offset is provided (must come before limit)
    if (q.offset) {
      pipeline.push({ $skip: q.offset });
    }

    // LIMIT: default 100, cap 1000
    const limit = Math.min(q.limit ?? 100, 1000);
    pipeline.push({ $limit: limit });

    const cursor = coll.aggregate(pipeline, { allowDiskUse: true });
    const rows = await cursor.toArray();
    return { rows, meta: { pipeline, rowCount: rows.length } };
  }

  async explain(q: ReadQuery) {
    // Simple placeholder (kept additive)
    return { note: 'TODO explain (Phase-3)', entity: q.entity };
  }

  async health() {
    try {
      await this.cli.db(this.dbName).command({ ping: 1 });
      return { ok: true };
    } catch (e: any) {
      return { ok: false, error: e?.message ?? String(e) };
    }
  }
}

export default MongoAdapter;