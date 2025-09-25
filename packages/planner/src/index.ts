// packages/planner/src/index.ts
// Phase-2.5 → 3 planner — deterministic + beam; joins (1-hop), aggregates,
// LIKE/CONTAINS/IS passthrough; adds detail + compare (multi-target + period).
import type { IQL, UQL, UEM, IEM, ProofTraceV2 } from '@nauvra/core';
import { IQLSchema } from '@nauvra/core';
import { loadIEM } from '@nauvra/uem';
import {
  alignmentScore, pickTopK, softmaxPick, totalScore,
  slotCostPenalty, computeCostSketch
} from './scorer.js';

// ---------- tiny helpers ----------
function simTokens(a: string, b: string): number {
  const tok = (s:string)=> new Set((s||'').toLowerCase().split(/[^a-z0-9]+/g).filter(Boolean));
  const A=tok(a), B=tok(b); const inter=[...A].filter(x=>B.has(x)).length;
  return inter/Math.max(1, Math.max(A.size,B.size));
}
function getEntity(uem: UEM, name: string){ return uem.entities.find(e=>e.name===name); }
function fieldRole(uem: UEM, entity: string, fieldName: string): string | undefined {
  const ent = getEntity(uem, entity);
  return (ent?.fields.find(f=>f.name===fieldName) as any)?.role;
}
function fieldVec(iem: IEM | null, entity: string, field: string): number[] | undefined {
  if (!iem) return undefined;
  return iem.fields.find(x => x.entity === entity && x.name === field)?.vec;
}
function fieldAlign(entity: string, fieldName: string, slotLabel: string, opts: { iem?: IEM|null; intentVec?: number[]; fieldRole?: string }){
  const { iem, intentVec, fieldRole: role } = opts;
  const vec = fieldVec(iem ?? null, entity, fieldName);
  if (vec && intentVec && intentVec.length && vec.length) {
    const cos = alignmentScore(vec, intentVec);
    const bonus = role && /(city|country|state|geo|location)/i.test(slotLabel) && role==='geo' ? 0.1 : 0;
    return cos + bonus;
  }
  return simTokens(slotLabel, fieldName) + (role==='geo' && /(city|country)/i.test(slotLabel) ? 0.1 : 0);
}
function enumerateSlotCandidates(uem: UEM, entity: string, fields: string[], slotLabel: string, iem: IEM|null, intentVec?: number[]){
  const cands = fields.map(name => ({
    value: name,
    align: fieldAlign(entity, name, slotLabel, { iem, intentVec, fieldRole: fieldRole(uem, entity, name) }),
    source: iem ? ('iem' as const) : ('heuristic' as const)
  }));
  return pickTopK(cands, 8, c=>c.align);
}
function dateTruncExpr(grain: 'day'|'week'|'month', field='created_at'){ return `date_trunc_${grain}(${field})`; }
function lastSegment(qualified: string){ return qualified.split('.').pop() || qualified; }

// ---------- JOIN inference (1-hop) ----------
function inferJoinForQualifiedTarget(uem: UEM, base: string, targetExpr: string):
  { joins?: UQL['joins'], groupByExpr: string, selectExpr: string } {
  if (!targetExpr.includes('.')) return { groupByExpr: targetExpr, selectExpr: targetExpr };

  const [toEntity] = targetExpr.split('.', 2);
  const baseEnt = uem.entities.find(e => e.name === base);
  if (!baseEnt) return { groupByExpr: targetExpr, selectExpr: targetExpr };

  const fk = baseEnt.fields.find(f => f.ref?.entity === toEntity);
  if (!fk) return { groupByExpr: targetExpr, selectExpr: targetExpr };

  const joins: UQL['joins'] = [{
    type: 'inner',
    from: base,
    to: toEntity,
    on: { left: fk.name, right: (uem.entities.find(e=>e.name===toEntity)?.primaryKey || 'id') }
  }];

  return { joins, groupByExpr: targetExpr, selectExpr: targetExpr };
}

// ---------- metric helper (adds count_distinct) ----------
// NOTE: we strip the qualifier from metric.over for now. If we later allow
// metric over qualified fields (e.g. "customer.id"), we'll need join inference
// for metrics as well (not just targets).
function metricExpr(iql: IQL): { expr: string; alias: string } {
  const op = (iql.metric?.op ?? 'count') as ('count'|'sum'|'avg'|'min'|'max'|'count_distinct');
  const overRaw = iql.metric?.over || 'order.id';
  const over = overRaw.split('.').pop() || '*';
  const alias = iql.metric?.name ?? (op === 'count' ? 'count' : `${op}_${over}`);

  if (op === 'count') return { expr: 'count(*)', alias: alias === 'count' ? 'order_count' : alias };
  if (op === 'count_distinct') return { expr: `count_distinct(${over})`, alias: alias || `count_distinct_${over}` };
  return { expr: `${op}(${over})`, alias };
}

// ---------- orderBy helper (multi-field) ----------
function buildOrderBy(iql: IQL, fallback?: {field:string,dir:'asc'|'desc'}[]): {field:string,dir:'asc'|'desc'}[]|undefined {
  if (iql.orderBy && iql.orderBy.length) return iql.orderBy as any;
  return fallback;
}

// ---------- paging helper (limit + offset | cursor) ----------
function getPaging(iql: IQL): { limit?: number; offset?: number } {
  const limit = iql.limit ?? undefined;
  let offset: number | undefined = undefined;

  // Support either `offset` (if user passes it) or numeric cursor (back-compat)
  const anyIQL = iql as any;
  if (typeof anyIQL.offset === 'number' && anyIQL.offset >= 0) offset = anyIQL.offset;
  else if (iql.cursor && /^\d+$/.test(String(iql.cursor))) offset = parseInt(String(iql.cursor), 10);

  return { limit, offset };
}

// -----------------------------
// Compiler (deterministic + beam)
// -----------------------------
export function compileIQLtoUQL(
  iql: IQL,
  uem: UEM,
  opts?: { intentVec?: number[]; plannerMode?: 'deterministic'|'beam'; tau?: number; beamWidth?: number }
): { uql: UQL; trace: ProofTraceV2 } {
  // Keep schema validation; it tolerates unknown fields (e.g., targets, comparePeriods)
  IQLSchema.parse(iql);

  const iem = (loadIEM() as IEM) || null;
  const intentVec = opts?.intentVec;
  const mode = opts?.plannerMode ?? ((process.env.PLANNER_MODE as any) || 'deterministic');
  const tau = opts?.tau ?? 0.7;
  const beamWidth = opts?.beamWidth ?? 4;

  const entity = 'sales_order';
  const ent = getEntity(uem, entity);
  if (!ent) throw new Error(`Entity not found in UEM: ${entity}`);
  const allFields = ent.fields.map(f=>f.name);

  // ---------- TREND ----------
  if (iql.ask === 'trend') {
    const grain = (iql as any).grain ?? 'day';
    const expr = dateTruncExpr(grain, 'created_at');
    const { expr: mexpr, alias: malias } = metricExpr({ ...iql, metric: { op: (iql.metric?.op ?? 'count') as any, over: iql.metric?.over ?? 'order.id', name: iql.metric?.name ?? 'order_count' } });

    const and: any[] = [
      ...(iql.filters ?? []).map(f => ({
        field: String(f.field || '').replace(/^order\./,''), op: f.op as any, value: f.value
      })),
      ...(iql.timeWindow ? [{ field: 'created_at', op: 'between', value: [iql.timeWindow.start, iql.timeWindow.end] }] : [])
    ];

    const { limit } = getPaging(iql);

    const uql: UQL = {
      entity,
      select: [{ expr, as: 'bucket' }, { expr: mexpr, as: malias }],
      where: and.length ? { and } : undefined,
      groupBy: ['bucket'],
      orderBy: buildOrderBy(iql, [{ field: 'bucket', dir: 'asc' }]),
      limit,
      version: 'uql/0.1',
      hints: { requireIndexedFilters: true }
    };
    const cost = computeCostSketch(uem, entity, { target: 'bucket', filter1: 'created_at', hasTimeWindow: !!iql.timeWindow });
    return { uql, trace: { joins: [], cost, planner: { mode } } };
  }

  // ---------- DETAIL (drilldown) ----------
  if (iql.ask === 'detail') {
    const { limit, offset } = getPaging(iql);
    const and: any[] = [
      ...(iql.filters ?? []).map(f => ({
        field: String(f.field || '').replace(/^order\./,''), op: f.op as any, value: f.value
      })),
      ...(iql.timeWindow ? [{ field: 'created_at', op: 'between', value: [iql.timeWindow.start, iql.timeWindow.end] }] : [])
    ];

    const defaultSelect = [
      { expr: 'id', as: 'id' },
      { expr: 'customer_id', as: 'customer_id' },
      { expr: 'total_amount', as: 'total_amount' },
      { expr: 'country', as: 'country' },
      { expr: 'shipping_city', as: 'city' },
      { expr: 'created_at', as: 'created_at' }
    ] as const;
    const explicitSelect = Array.isArray((iql as any).select) && (iql as any).select.length
      ? (iql as any).select
      : defaultSelect;

    const uql: UQL = {
      entity,
      select: explicitSelect as any,
      where: and.length ? { and } : undefined,
      orderBy: buildOrderBy(iql, [{ field: 'created_at', dir: 'desc' }]),
      limit: limit ?? 50,
      offset,
      version: 'uql/0.1',
      hints: { requireIndexedFilters: true }
    };
    const cost = computeCostSketch(uem, entity, { target: 'created_at', filter1: 'created_at', hasTimeWindow: !!iql.timeWindow });
    return { uql, trace: { joins: [], cost, planner: { mode } } };
  }

  // ---------- COMPARE ----------
  if (iql.ask === 'compare') {
    const anyIQL = iql as any;

    // targets: support either iql.targets (array) or comma-separated iql.target
    const targets: string[] =
      Array.isArray(anyIQL.targets) && anyIQL.targets.length
        ? anyIQL.targets.map((t:string)=>String(t))
        : (iql.target && iql.target.includes(','))
          ? iql.target.split(',').map(s=>s.trim()).filter(Boolean)
          : (iql.target ? [iql.target] : []);

    // period windows: non-strict — planner accepts, adapters will implement union/switch later
    const comparePeriods: {label:string,start:string,end:string}[] =
      Array.isArray(anyIQL.comparePeriods) ? anyIQL.comparePeriods : [];

    // metric (supports count_distinct too)
    const { expr: mexpr, alias: malias } = metricExpr(iql);

    // base filters (excluding timeWindow if comparePeriods present)
    const and: any[] = [
      ...(iql.filters ?? []).map(f => ({
        field: String(f.field || '').replace(/^order\./,''), op: f.op as any, value: f.value
      })),
      ...(!comparePeriods.length && iql.timeWindow
          ? [{ field: 'created_at', op: 'between', value: [iql.timeWindow.start, iql.timeWindow.end] }]
          : [])
    ];

    // ----- compare by target(s) only (multi-groupBy) -----
    if (!comparePeriods.length) {
      // infer join if any target is qualified
      let joins: UQL['joins'] | undefined = undefined;
      const gb: string[] = [];
      const sel: {expr:string,as?:string}[] = [];

      for (const t of targets) {
        const { joins: j2, groupByExpr, selectExpr } = inferJoinForQualifiedTarget(uem, entity, t);
        if (j2) {
          joins = (joins || []).concat(j2) as any;
          const seen = new Set<string>();
          joins = (joins || []).filter((j: any) => (seen.has(j.to) ? false : (seen.add(j.to), true)));
        }
        gb.push(groupByExpr);
        sel.push({ expr: selectExpr, as: lastSegment(t) });
      }

      // default: if no targets, fallback to a single bucket by day (so compare has a shape)
      if (!gb.length) {
        const bucket = dateTruncExpr(((iql as any).grain ?? 'day') as any, 'created_at');
        gb.push(bucket);
        sel.push({ expr: bucket, as: 'bucket' });
      }

      sel.push({ expr: mexpr, as: malias });

      const uql: UQL = {
        entity,
        joins,
        select: sel,
        where: and.length ? { and } : undefined,
        groupBy: gb,
        orderBy: buildOrderBy(iql, [{ field: malias, dir: 'desc' }]),
        limit: iql.limit ?? undefined,
        offset: (iql as any).offset ?? undefined,
        version: 'uql/0.1',
        hints: { requireIndexedFilters: true }
      };
      const cost = computeCostSketch(uem, entity, { target: gb.join(','), filter1: 'created_at', hasTimeWindow: !!iql.timeWindow });
      return { uql, trace: { joins: joins ?? [], cost, planner: { mode } } };
    }

    // ----- compare by period windows (with labels) -----
    // We project a virtual "period" label and group by it (and any targets).
    // Adapters will map `period_label(created_at)` using hints.comparePeriods.
    let joins: UQL['joins'] | undefined = undefined;
    const gb: string[] = ['__period__'];    // normalized key for adapters
    const sel: {expr:string,as?:string}[] = [{ expr: 'period_label(created_at)', as: 'period' }];

    for (const t of targets) {
      const { joins: j2, groupByExpr, selectExpr } = inferJoinForQualifiedTarget(uem, entity, t);
      if (j2) {
        joins = (joins || []).concat(j2) as any;
        const seen = new Set<string>();
        joins = (joins || []).filter((j: any) => (seen.has(j.to) ? false : (seen.add(j.to), true)));
      }
      gb.push(groupByExpr);
      sel.push({ expr: selectExpr, as: lastSegment(t) });
    }

    sel.push({ expr: mexpr, as: malias });

    const uql: UQL = {
      entity,
      joins,
      select: sel,
      where: and.length ? { and } : undefined,
      groupBy: gb,
      orderBy: buildOrderBy(iql, [{ field: 'period', dir: 'asc' }]),
      limit: iql.limit ?? undefined,
      offset: (iql as any).offset ?? undefined,
      version: 'uql/0.1',
      // pass the periods to adapters via hints; they will expand to UNION/$unionWith or CASE/$switch
      hints: { requireIndexedFilters: true, comparePeriods }
    };
    const cost = computeCostSketch(uem, entity, { target: gb.join(','), filter1: 'created_at', hasTimeWindow: true });
    return { uql, trace: { joins: joins ?? [], cost, planner: { mode } } };
  }

  // ---------- TOP_K (deterministic) ----------
  if (mode !== 'beam') {
    const requestedTarget = iql.target ?? 'city';

    if (requestedTarget.includes('.')) {
      const { joins, groupByExpr, selectExpr } =
        inferJoinForQualifiedTarget(uem, entity, requestedTarget);

      const { expr: mexpr, alias: malias } = metricExpr(iql);
      const and: any[] = [];

      for (const f of (iql.filters ?? [])) {
        const field = String(f.field || '').replace(/^order\./,'');
        and.push({ field, op: f.op as any, value: f.value as any });
      }
      if (iql.timeWindow) {
        and.push({ field: 'created_at', op: 'between', value: [iql.timeWindow.start, iql.timeWindow.end] });
      }

      const uql: UQL = {
        entity,
        joins,
        select: [
          { expr: selectExpr, as: requestedTarget.split('.').pop() || 'group' },
          { expr: mexpr, as: iql.metric?.name ?? (malias || 'order_count') }
        ],
        where: and.length ? { and } : undefined,
        groupBy: [groupByExpr],
        orderBy: buildOrderBy(iql, [{ field: iql.metric?.name ?? (malias || 'order_count'), dir: 'desc' }]),
        limit: iql.k ?? iql.limit ?? 10,
        version: 'uql/0.1',
        hints: { requireIndexedFilters: true }
      };

      const cost = computeCostSketch(uem, entity, {
        target: requestedTarget,
        filter0: (iql.filters?.[0]?.field || 'country').replace(/^order\./,''),
        filter1: 'created_at',
        hasTimeWindow: !!iql.timeWindow
      });

      return { uql, trace: { joins: joins ?? [], cost, planner: { mode: 'deterministic' } } };
    }

    // fallback ranking over base fields
    const ranked = allFields
      .map(name => ({ name, score: simTokens(requestedTarget, name) + (fieldRole(uem, entity, name)==='geo'?0.2:0) }))
      .sort((a,b)=>b.score-a.score);
    const target = ranked[0]?.name ?? 'shipping_city';
    const { expr: mexpr, alias: malias } = metricExpr(iql);

    const { joins, groupByExpr, selectExpr } =
      inferJoinForQualifiedTarget(uem, entity, target);

    const select = [
      { expr: selectExpr, as: 'city' },
      { expr: mexpr, as: iql.metric?.name ?? (malias || 'order_count') }
    ];

    const groupBy = [groupByExpr];

    const and: any[] = [];
    for (const f of (iql.filters ?? [])) {
      const field = String(f.field || '').replace(/^order\./,'');
      and.push({ field, op: f.op as any, value: f.value as any });
    }
    if (iql.timeWindow) and.push({ field: 'created_at', op: 'between', value: [iql.timeWindow.start, iql.timeWindow.end] });

    if (!target.includes('.')) {
      and.push({ field: target, op: 'exists', value: true });
    }

    const uql: UQL = {
      entity,
      joins,
      select,
      where: { and },
      groupBy,
      orderBy: buildOrderBy(iql, [{ field: iql.metric?.name ?? (malias || 'order_count'), dir: 'desc' }, { field: 'city', dir: 'asc' }]),
      limit: iql.k ?? iql.limit ?? 10,
      version: 'uql/0.1',
      hints: { requireIndexedFilters: true }
    };
    const cost = computeCostSketch(uem, entity, { target, filter0: 'country', filter1: 'created_at', hasTimeWindow: !!iql.timeWindow });
    return { uql, trace: { joins: joins ?? [], cost, planner: { mode: 'deterministic' } } };
  }

  // ---------- TOP_K (BEAM MODE) ----------
  if ((iql.target ?? '').includes('.')) {
    const requestedTarget = iql.target as string;
    const { joins, groupByExpr, selectExpr } =
      inferJoinForQualifiedTarget(uem, entity, requestedTarget);
    const { expr: mexpr, alias: malias } = metricExpr(iql);

    const providedFilters = (iql.filters ?? []).map(f => ({
      field: String(f.field || '').replace(/^order\./, ''),
      op: f.op as any,
      value: f.value
    }));
    if (iql.timeWindow) {
      providedFilters.push({
        field: 'created_at',
        op: 'between' as const,
        value: [iql.timeWindow.start, iql.timeWindow.end]
      });
    }

    const uql: UQL = {
      entity,
      joins,
      select: [
        { expr: selectExpr, as: requestedTarget.split('.').pop() || 'group' },
        { expr: mexpr, as: iql.metric?.name ?? (malias || 'order_count') }
      ],
      where: providedFilters.length ? { and: providedFilters } : undefined,
      groupBy: [groupByExpr],
      orderBy: buildOrderBy(iql, [{ field: iql.metric?.name ?? (malias || 'order_count'), dir: 'desc' }]),
      limit: iql.limit ?? iql.k ?? 10,
      version: 'uql/0.1',
      hints: { requireIndexedFilters: true }
    };

    const cost = computeCostSketch(uem, entity, {
      target: requestedTarget, filter0: providedFilters[0]?.field, filter1: 'created_at', hasTimeWindow: !!iql.timeWindow
    });

    return { uql, trace: { joins: joins ?? [], cost, policy: {}, planner: { mode: 'beam', tau, beamWidth } } };
  }

  const targetLabel = iql.target ?? 'city';
  const metricOverLabel = (iql.metric?.over || 'order.id').split('.').pop() || 'id';
  const filter0Label = (iql.filters?.[0]?.field || 'order.country').replace(/^order\./, '');
  const filter1Label = iql.timeWindow ? 'created_at' : ((iql.filters?.[1]?.field || 'created_at').replace(/^order\./, ''));

  const slots = [
    { name: 'target', candidates: enumerateSlotCandidates(uem, entity, allFields, targetLabel, iem, intentVec) },
    { name: 'metric.over', candidates: enumerateSlotCandidates(uem, entity, allFields, metricOverLabel, iem, intentVec) },
    { name: 'filter.field[0]', candidates: enumerateSlotCandidates(uem, entity, allFields, filter0Label, iem, intentVec) },
    { name: 'filter.field[1]', candidates: enumerateSlotCandidates(uem, entity, allFields, filter1Label, iem, intentVec) }
  ] as const;

  type PartialPlan = { slots: Record<string, string | undefined>; score: number; steps: any[] };
  let beam: PartialPlan[] = [{ slots: {}, score: 0, steps: [] }];
  const hasTimeWindow = !!iql.timeWindow;

  for (const slot of slots) {
    const expanded: PartialPlan[] = [];
    for (const st of beam) {
      for (const c of slot.candidates) {
        const align = c.align;
        const pen = slotCostPenalty(uem, entity, slot.name, c.value, { hasTimeWindow });
        const parsimony = 0;
        const delta = totalScore(align, pen.penalty, parsimony).total;
        expanded.push({
          slots: { ...st.slots, [slot.name]: c.value },
          score: st.score + delta,
          steps: [...st.steps, { slot: slot.name, value: c.value, breakdown: { align, cost: pen.penalty, parsimony, delta } }]
        });
      }
    }
    const picked: PartialPlan[] = [];
    const pool = expanded.slice();
    const B = Math.min(beamWidth, pool.length);
    for (let i=0;i<B;i++){
      const chosen = softmaxPick(pool, x=>x.score, tau);
      picked.push(chosen);
      const idx = pool.indexOf(chosen);
      if (idx >= 0) pool.splice(idx,1);
    }
    beam = picked;
  }

  const best = beam[0] || { slots: {}, score: 0, steps: [] };
  const target = (best.slots['target'] as string) ?? 'shipping_city';
  const filter0 = (best.slots['filter.field[0]'] as string) ?? 'country';
  const filter1 = (best.slots['filter.field[1]'] as string) ?? 'created_at';
  const { expr: mexpr, alias: malias } = metricExpr(iql);

  const { joins, groupByExpr, selectExpr } =
    inferJoinForQualifiedTarget(uem, entity, target);

  const select = [
    { expr: selectExpr, as: 'city' },
    { expr: mexpr, as: iql.metric?.name ?? (malias || 'order_count') }
  ];

  const groupBy = [groupByExpr];

  const providedFilters = (iql.filters ?? []).map(f => ({
    field: String(f.field || '').replace(/^order\./, ''),
    op: f.op as any,
    value: f.value
  }));
  if (filter0) {
    if (providedFilters[0]) providedFilters[0].field = filter0;
    else providedFilters.unshift({ field: filter0, op: (iql.filters?.[0]?.op as any) || 'eq', value: iql.filters?.[0]?.value ?? 'IN' });
  }
  if (iql.timeWindow) {
    providedFilters.push({ field: filter1, op: 'between' as const, value: [iql.timeWindow.start, iql.timeWindow.end] });
  }
  if (!target.includes('.')) {
    providedFilters.push({ field: target, op: 'exists' as const, value: true });
  }

  const uql: UQL = {
    entity,
    joins,
    select,
    where: { and: providedFilters },
    groupBy,
    orderBy: buildOrderBy(iql, [{ field: iql.metric?.name ?? (malias || 'order_count'), dir: 'desc' }, { field: 'city', dir: 'asc' }]),
    limit: iql.limit ?? iql.k ?? 10,
    version: 'uql/0.1',
    hints: { requireIndexedFilters: true }
  };

  const slotCandidates = slots.flatMap(s =>
    s.candidates.slice(0, 5).map(c => ({
      slot: s.name, value: c.value, score: c.align,
      breakdown: { align: c.align, cost: 0, parsimony: 0 },
      source: c.source
    }))
  );
  const cost = computeCostSketch(uem, entity, { target, filter0, filter1, hasTimeWindow });
  return { uql, trace: { slotCandidates, joins: joins ?? [], cost, policy: {}, planner: { mode: 'beam', tau, beamWidth } } };
}