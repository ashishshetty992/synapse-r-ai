// packages/uql-mysql/src/index.ts
import { Kysely, MysqlDialect } from 'kysely';
import mysql, { Pool } from 'mysql2/promise';
import type { Adapter, ReadQuery, UEM } from '@nauvra/core';
import { requireIndexed } from '@nauvra/core';

export class MySQLAdapter implements Adapter {
  name: 'mysql' = 'mysql';
  private db!: Kysely<any>;
  public  pool!: Pool;       // public so /seed can use it
  private uem!: UEM;

  async init(uem: UEM, cfg: { uri?: string } = {}) {
    this.uem = uem;
    this.pool = mysql.createPool(cfg.uri ?? 'mysql://root:root@127.0.0.1:3306/nauvra');
    this.db = new Kysely({ dialect: new MysqlDialect({ pool: this.pool }) });
  }

  // --- helpers (additive) ---
  private qid(x: string): string {
    // backtick-escape identifiers (simple, safe)
    return `\`${x}\``;
  }
  private wrapTZ(colSQL: string, tzOverride?: string): string {
    const tz = tzOverride ?? process.env.TIMEZONE;
    return tz ? `CONVERT_TZ(${colSQL}, 'UTC', ${mysql.escape(tz)})` : colSQL;
  }
  private weekStartMode(weekStartOverride?: string): 'mon'|'sun' {
    const w = (weekStartOverride ?? process.env.WEEK_START ?? 'mon').toLowerCase();
    return (w === 'sun') ? 'sun' : 'mon';
  }

  // Build table->alias map: base t0, each join t1, t2, ...
  private buildAliasMap(q: ReadQuery) {
    const map: Record<string,string> = {};
    map[q.entity] = 't0';
    (q.joins ?? []).forEach((j, i) => { map[j.to] = `t${i+1}`; });
    return map;
  }

  private qualify(col: string, aliasMap: Record<string,string>, baseAlias: string): string {
    // Qualify a column/field path with table alias; quote field/identifier
    if (col.includes('.')) {
      const [tbl, field] = col.split('.', 2);
      const a = aliasMap[tbl] ?? tbl; // if not joined, leave as-is (defensive)
      return `${a}.${this.qid(field)}`;
    }
    return `${baseAlias}.${this.qid(col)}`;
  }

  // Map UQL expr → MySQL SQL expr, with aliasing
  private exprToSQL(expr: string, aliasMap: Record<string,string>, baseAlias: string, hints?: any): string {
    // count_distinct(...)
    const mCD = expr.match(/^count_distinct\(([^)]+)\)$/i);
    if (mCD) {
      const inner = mCD[1].trim();
      const qualified = /^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)?$/.test(inner)
        ? this.qualify(inner, aliasMap, baseAlias)
        : inner;
      return `COUNT(DISTINCT ${qualified})`;
    }

    // date_trunc_* with TZ + WEEK_START handling
    const m = expr.match(/^date_trunc_(day|week|month)\(([^)]+)\)$/i);
    if (m) {
      const unit = m[1].toLowerCase();
      const raw = m[2];
      const col0 = this.qualify(raw, aliasMap, baseAlias);
      const col = this.wrapTZ(col0, hints?.timezone);
      if (unit === 'day')   return `DATE(${col})`;
      if (unit === 'week')  {
        const wkMode = this.weekStartMode(hints?.weekStart);
        const wkArg = wkMode === 'sun' ? 0 : 3; // 0 → Sunday, 3 → ISO Monday
        const lbl   = wkMode === 'sun' ? 'Sunday' : 'Monday';
        return `STR_TO_DATE(CONCAT(YEARWEEK(${col}, ${wkArg}), ' ${lbl}'), '%X%V %W')`;
      }
      if (unit === 'month') return `DATE_FORMAT(${col}, '%Y-%m-01')`;
    }

    if (/^count\(\*\)$/i.test(expr)) return 'count(*)';

    // plain identifier or qualified identifier
    if (/^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)?$/.test(expr)) {
      return this.qualify(expr, aliasMap, baseAlias);
    }

    // passthrough for raw functions etc. (function allowlist could be enforced here if desired)
    return expr;
  }

  // Heuristic: collect qualified refs like "customer.segment" across select/group/order/where
  private collectQualifiedRefs(q: ReadQuery): Set<string> {
    const refs = new Set<string>();
    const see = (s?: string) => {
      if (!s) return;
      const m = s.match(/^([a-zA-Z_][a-zA-Z0-9_]*)\.[a-zA-Z_][a-zA-Z0-9_]*$/);
      if (m) refs.add(m[1]); // table/entity prefix
    };
    for (const s of q.select ?? []) see(s.expr);
    for (const g of q.groupBy ?? []) see(g);
    for (const o of q.orderBy ?? []) see(o.field);
    const walk = (w: any) => {
      if (!w) return;
      if ('and' in w) return (w.and ?? []).forEach(walk);
      if ('or'  in w) return (w.or  ?? []).forEach(walk);
      if (w.field) see(w.field);
    };
    if (q.where) walk(q.where);
    return refs;
  }

  // If planner forgot a join (e.g. qualified "customer.segment"), synthesize INNER JOIN:
  //   ON <base>.<customer_id> = customer.id
  private synthesizeMissingJoins(q: ReadQuery): ReadQuery {
    const base = q.entity;
    const declared = new Set((q.joins ?? []).map(j => j.to));
    const needed = this.collectQualifiedRefs(q);
    needed.delete(base);
    const extras = [...needed]
      .filter(t => !declared.has(t))
      .map(to => ({
        type: 'inner' as const,
        from: base,
        to,
        on: { left: `${to}_id`, right: 'id' }
      }));
    return extras.length ? { ...q, joins: [...(q.joins ?? []), ...extras] } : q;
  }

  // Build CASE for period_label(created_at) from hints.comparePeriods
  private buildPeriodCaseExpr(baseAlias: string, comparePeriods?: Array<{label:string,start:string,end:string}>): string | null {
    if (!comparePeriods || !comparePeriods.length) return null;
    const col = `${baseAlias}.${this.qid('created_at')}`;
    const parts = comparePeriods.map(p =>
      `WHEN ${col} >= ${mysql.escape(p.start)} AND ${col} < ${mysql.escape(p.end)} THEN ${mysql.escape(p.label)}`
    ).join(' ');
    return `CASE ${parts} ELSE 'Other' END`;
  }

  async read(q: ReadQuery) {
    if (q.hints?.requireIndexedFilters) requireIndexed(this.uem, q.entity, q.where);

    // Heuristic backstop: add missing joins if qualified refs appear
    const q2 = this.synthesizeMissingJoins(q);

    const base = q2.entity;
    const aliasMap = this.buildAliasMap(q2);
    const baseAlias = aliasMap[base];

    // Build period CASE expression if comparePeriods are present
    const periodCase = this.buildPeriodCaseExpr(baseAlias, (q2 as any).hints?.comparePeriods);

    // build a quick alias lookup
    const selectAliasToExpr = new Map<string, string>();
    for (const s of q2.select) {
      if (s.as) selectAliasToExpr.set(s.as, s.expr);
    }
    const hasSelectAlias = (name: string) => selectAliasToExpr.has(name);

    // SELECT
    const sel = q2.select.map((s) => {
      // Support planner's virtual "period_label(created_at)"
      if (/^period_label\s*\(\s*created_at\s*\)$/i.test(s.expr) && periodCase) {
        return `${periodCase}${s.as ? ` AS ${s.as}` : ''}`;
      }
      const e = this.exprToSQL(s.expr, aliasMap, baseAlias, q2.hints);
      return `${e}${s.as ? ` AS ${s.as}` : ''}`;
    }).join(', ');

    // FROM + JOINs
    let sqlText = `SELECT ${sel} FROM ${this.qid(base)} ${baseAlias}`;
    (q2.joins ?? []).forEach((j, i) => {
      const alias = aliasMap[j.to]; // t1, t2, ...
      const left  = this.qualify(j.on.left, aliasMap, baseAlias);
      const right = this.qualify(`${j.to}.${j.on.right}`, aliasMap, baseAlias);
      const jt = (j.type === 'left') ? 'LEFT JOIN' : 'INNER JOIN';
      sqlText += ` ${jt} ${this.qid(j.to)} ${alias} ON ${left} = ${right}`;
    });

    // WHERE
    const params: any[] = [];
    const leaf = (w: any): string => {
      if ('and' in w) return '(' + w.and.map(leaf).join(' AND ') + ')';
      if ('or'  in w) return '(' + w.or.map(leaf).join(' OR ') + ')';
      const col = this.qualify(w.field, aliasMap, baseAlias);
      switch (w.op) {
        case 'eq': params.push(w.value); return `${col} = ?`;
        case 'neq': params.push(w.value); return `${col} <> ?`;
        case 'between': params.push(w.value[0], w.value[1]); return `${col} >= ? AND ${col} < ?`;
        case 'exists': return w.value ? `${col} IS NOT NULL` : `${col} IS NULL`;
        case 'gt': params.push(w.value); return `${col} > ?`;
        case 'gte': params.push(w.value); return `${col} >= ?`;
        case 'lt': params.push(w.value); return `${col} < ?`;
        case 'lte': params.push(w.value); return `${col} <= ?`;
        case 'in':  { const arr = Array.isArray(w.value) ? w.value : [w.value]; params.push(...arr); return `${col} IN (${arr.map(()=>'?').join(',')})`; }
        case 'nin': { const arr = Array.isArray(w.value) ? w.value : [w.value]; params.push(...arr); return `${col} NOT IN (${arr.map(()=>'?').join(',')})`; }
        case 'like': {
          const raw = String(w.value ?? '');
          const hasWildcard = /[%_]/.test(raw);
          const escaped = raw.replace(/[%_]/g, m => '\\' + m);
          const patt = hasWildcard ? raw : `${escaped}%`; // default prefix
          params.push(patt);
          return `${col} LIKE ?`;
        }
        case 'contains': {
          // case-insensitive contains
          params.push(String(w.value ?? '').toLowerCase());
          return `LOWER(${col}) LIKE CONCAT('%', ?, '%')`;
        }
        case 'ilike': {
          // case-insensitive like (prefix match by default if no wildcard)
          const raw = String(w.value ?? '');
          const patt = /[%_]/.test(raw) ? raw : `${raw}%`;
          params.push(patt.toLowerCase());
          return `LOWER(${col}) LIKE ?`;
        }
        case 'is': {
          if (w.value === 'null')     return `${col} IS NULL`;
          if (w.value === 'not_null') return `${col} IS NOT NULL`;
          if (w.value === 'true')     return `${col} = TRUE`;
          if (w.value === 'false')    return `${col} = FALSE`;
          throw new Error(`Unsupported is-operand: ${w.value}`);
        }
        default: throw new Error(`Unsupported op in MySQL adapter: ${w.op}`);
      }
    };
    if (q2.where) sqlText += ` WHERE ${leaf(q2.where)}`;

    // --- auto GROUP BY for mixed selects (agg + non-agg) ---
    const isAgg = (e: string) =>
      /^(count\s*\(\s*\*?\s*\)|sum\s*\(|avg\s*\(|min\s*\(|max\s*\(|count_distinct\s*\()/i.test((e||'').trim());

    const hasAggInSelect = q2.select.some(s => isAgg(s.expr));
    let finalGroupBy = Array.isArray(q2.groupBy) ? [...q2.groupBy] : [];

    if (hasAggInSelect) {
      // ensure all non-agg projections appear in GROUP BY
      for (const s of q2.select) {
        if (!isAgg(s.expr)) {
          const key = s.as ?? s.expr;
          if (!finalGroupBy.includes(key)) finalGroupBy.push(key);
        }
      }
    }

    // GROUP BY (render only if we have keys)
    if (finalGroupBy.length > 0) {
      const groupParts = finalGroupBy.map(g => {
        if (g === '__period__' && periodCase) return periodCase;
        if (hasSelectAlias(g)) {
          const expr = selectAliasToExpr.get(g)!;
          if (/^period_label\s*\(\s*created_at\s*\)$/i.test(expr) && periodCase) {
            return periodCase;
          }
          return this.exprToSQL(expr, aliasMap, baseAlias, q2.hints);
        }
        return this.exprToSQL(g, aliasMap, baseAlias, q2.hints);
      });
      const uniqueParts = Array.from(new Set(groupParts));
      sqlText += ` GROUP BY ${uniqueParts.join(', ')}`;
    }

    // ORDER BY
    if (q2.orderBy?.length) {
      const orderSql = q2.orderBy.map(o => {
        // If the field is a SELECT alias, keep it verbatim (no qualification!)
        if (hasSelectAlias(o.field)) {
          return `${o.field} ${o.dir.toUpperCase()}`;
        }
        return `${this.exprToSQL(o.field, aliasMap, baseAlias, q2.hints)} ${o.dir.toUpperCase()}`;
      }).join(', ');
      sqlText += ` ORDER BY ${orderSql}`;
    }

    // Default limit + cap
    const cap = 1000;
    const effLimit = Math.min(q2.limit ?? 100, cap);
    sqlText += ` LIMIT ${effLimit}${q2.offset ? ` OFFSET ${q2.offset}` : ''}`;

    console.log('[MySQL] Executing query:', {
      sql: sqlText,
      params,
      entity: q2.entity,
      joins: q2.joins?.length || 0,
      selectAliases: Array.from(selectAliasToExpr.keys()),
      groupBy: finalGroupBy,
      orderBy: q2.orderBy
    });

    const [rows] = await this.pool.query(sqlText, params);
    
    console.log('[MySQL] Query result:', {
      rowCount: (rows as any[]).length,
      firstRow: (rows as any[])[0] || null
    });

    return { rows: rows as any[], meta: { sql: sqlText, params, rowCount: (rows as any[]).length } };
  }

  async explain(q: ReadQuery) {
    // Reuse read() to build the exact SQL, then EXPLAIN it
    const built = await this.read({ ...q, limit: undefined });
    const sql = built.meta.sql;
    const params = built.meta.params || [];
    const [rows] = await this.pool.query(`EXPLAIN FORMAT=JSON ${sql}`, params);
    return { sql, params, explain: rows };
  }

  async health() { return { ok: true }; }
}

export default MySQLAdapter;