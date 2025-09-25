// packages/planner/src/beam.ts

export interface Slot {
  name: string;            // e.g., "target" | "metric.over" | "filter.field[0]"
  candidates: { value: string; align: number; source: 'iem'|'heuristic' }[];
}

export interface PartialPlan {
  slots: Record<string, string|undefined>;  // chosen values for slots
  score: number;                            // running total
  steps: any[];                             // trace per expansion
}

export interface BeamConfig {
  width: number;  // B
  tau: number;    // temperature for softmax exploration
}

/**
 * NOTE: Kept exactly as in your current code.
 * Returns softmax weights for an array of numeric scores.
 */
export function softmaxWeights(scores: number[], tau: number): number[] {
  const invT = 1/Math.max(1e-6, tau);
  const ex = scores.map(s => Math.exp(s*invT));
  const Z = ex.reduce((a,b)=>a+b,0);
  return ex.map(x => x / (Z || 1));
}

/**
 * NOTE: Kept exactly as in your current code.
 * Expand beam with next slot using provided scoring function.
 */
export function expandBeam(
  state: PartialPlan,
  next: Slot,
  scoreFn: (cand: {value:string;align:number})=>number
): PartialPlan[] {
  const out: PartialPlan[] = [];
  for (const c of next.candidates) {
    const sc = scoreFn(c);
    out.push({
      slots: { ...state.slots, [next.name]: c.value },
      score: state.score + sc,
      steps: [...state.steps, { slot: next.name, value: c.value, delta: sc, align: c.align }]
    });
  }
  return out;
}

/* -------------------------------------------------------------------------- */
/*                         ADDITIVE HELPERS (NEW)                              */
/* -------------------------------------------------------------------------- */

/** Local type for selects to avoid cross-package imports. */
export type SelectItem = { expr: string; as?: string };

/**
 * Heuristic: detect if a select expression is an aggregate.
 * Recognizes: count(*), sum(), avg(), min(), max(), count_distinct().
 * Treats date_trunc_* as NON-aggregate (bucket/group key).
 */
export function isAggregateExpr(expr: string): boolean {
  const e = String(expr || '').trim().toLowerCase().replace(/\s+/g, ' ');
  if (e === 'count(*)') return true;
  if (/^(sum|avg|min|max)\s*\(/.test(e)) return true;
  if (/^count_distinct\s*\(/.test(e)) return true;
  // Bucketing functions are not aggregates—they usually belong in GROUP BY.
  if (/^date_trunc_(day|week|month)\s*\(/.test(e)) return false;
  return false;
}

/**
 * Detect non-aggregate select items (these must appear in GROUP BY if any aggregate is present).
 * Returns the original select items that are NOT aggregates.
 */
export function detectNonAggSelects(
  selects: SelectItem,
  // no default aggOps list here; we rely on isAggregateExpr for consistency
): SelectItem[]; // overload to keep TS happy for single arg
export function detectNonAggSelects(
  selects: SelectItem[],
): SelectItem[];
export function detectNonAggSelects(
  selects: SelectItem | SelectItem[],
): SelectItem[] {
  const arr = Array.isArray(selects) ? selects : [selects];
  return arr.filter(s => !isAggregateExpr(s.expr));
}

/**
 * Compose final GROUP BY list using:
 *  - any planner-provided groupBys (e.g., targets, bucket),
 *  - plus all non-aggregate select aliases/exprs (if aggregates are present).
 *
 * If there are NO aggregates, returns provided groupBy (or undefined) unchanged.
 */
export function composeGroupBy(
  provided: string[] | undefined,
  selects: SelectItem[]
): string[] | undefined {
  const base = Array.isArray(provided) ? [...provided] : [];
  const hasAgg = selects.some(s => isAggregateExpr(s.expr));
  if (!hasAgg) {
    // Pure detail projection (no aggregates): GROUP BY not required.
    return base.length ? base : undefined;
  }

  // Enforce: include all non-aggregate projections in GROUP BY.
  for (const s of selects) {
    if (!isAggregateExpr(s.expr)) {
      const key = s.as || s.expr; // adapters can map alias→expr as needed
      if (!base.includes(key)) base.push(key);
    }
  }
  return base;
}

/**
 * Tiny helper to attach planner flags into a trace object (non-invasive).
 * Actual timing/metrics should be done in HTTP layer; this only tags flags.
 */
export function makePlannerTraceFlags(opts: { shortCircuit?: boolean; qualifiedTarget?: boolean }) {
  return {
    shortCircuit: !!opts.shortCircuit,
    qualifiedTarget: !!opts.qualifiedTarget,
  };
}