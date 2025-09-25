// packages/planner/src/scorer.ts
// Phase-2 scorer utilities: cosine, softmax(τ), penalties, cost sketch

import type { UEM } from '@nauvra/core';

// ----------------------
// Alignment (cosine) with guards
// ----------------------
export function alignmentScore(a?: number[], b?: number[]): number {
  if (!a || !b || a.length === 0 || b.length === 0) return 0;
  const n = Math.min(a.length, b.length);
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < n; i++) {
    const x = a[i], y = b[i];
    dot += x * y; na += x * x; nb += y * y;
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb) || 1;
  return dot / denom;
}

// ----------------------
// Simple top-k by key
// ----------------------
export function pickTopK<T>(arr: T[], k: number, score: (x: T) => number): T[] {
  return arr.slice().sort((a, b) => score(b) - score(a)).slice(0, k);
}

// ----------------------
// Softmax(τ) sampler (deterministic argmax over probs right now)
// Flip to stochastic by sampling from the CDF if desired.
// ----------------------
export function softmaxPick<T>(items: T[], score: (x: T) => number, tau = 0.7): T {
  if (items.length === 0) throw new Error('softmaxPick on empty set');
  const xs = items.map(score);
  const mx = Math.max(...xs);
  const exps = xs.map(s => Math.exp((s - mx) / Math.max(1e-6, tau)));
  const Z = exps.reduce((a, b) => a + b, 0) || 1;
  let best = 0, bi = 0;
  for (let i = 0; i < exps.length; i++) {
    const p = exps[i] / Z;
    if (p > best) { best = p; bi = i; }
  }
  return items[bi];
}

// ----------------------
// Score breakdown (α=1, β=1, γ=1 in v0)
// total = α·align − β·cost − γ·parsimony
// ----------------------
export function totalScore(align: number, costPenalty: number, parsimony: number) {
  const alpha = 1, beta = 1, gamma = 1;
  const total = alpha * align - beta * costPenalty - gamma * parsimony;
  return { total, alpha, beta, gamma };
}

// ----------------------
// Slot-level penalties: gently nudge away from risky choices
// ----------------------
export function slotCostPenalty(
  uem: UEM,
  entity: string,
  slot: string,
  field: string,
  ctx: { hasTimeWindow: boolean }
) {
  const ent = uem.entities.find(e => e.name === entity);
  const f = ent?.fields.find(ff => ff.name === field);
  if (!f) return { penalty: 0, reason: 'unknown-field' };

  // Time-window (range) wants an indexed timestamp-ish field
  if (slot === 'filter.field[1]' && ctx.hasTimeWindow) {
    const role = (f as any).role;
    const wantsTime = role === 'timestamp' || /(_at|date|time|ts)$/i.test(f.name);
    const idx = !!f.index;
    if (!wantsTime) return { penalty: 0.4, reason: 'not-timestamp' };
    if (!idx) return { penalty: 0.2, reason: 'no-index-on-time' };
  }

  // Generic nudge: filter slots prefer indexed fields
  if (/^filter\.field/.test(slot) && !f.index) {
    return { penalty: 0.1, reason: 'no-index' };
  }
  return { penalty: 0, reason: 'ok' };
}

// ----------------------
// Final cost sketch for ProofTrace
// Returns indexHits, estRows, and **structured** penalties
// ----------------------
export function computeCostSketch(
  uem: UEM,
  entity: string,
  choice: { target?: string; filter0?: string; filter1?: string; hasTimeWindow?: boolean }
) {
  const ent = uem.entities.find(e => e.name === entity);
  const rowCount = ent?.rowCount ?? 10000;

  const f0 = ent?.fields.find(f => f.name === choice.filter0); // eq filter (e.g., country)
  const f1 = ent?.fields.find(f => f.name === choice.filter1); // time range (e.g., created_at)

  // Selectivity heuristics
  const selF0 = (f0?.ndv && f0.ndv > 0) ? Math.min(1, 1 / Math.max(1, f0.ndv)) : 0.1;
  const selF1 = choice.hasTimeWindow ? 0.1 : 1.0;
  const estRows = Math.max(1, Math.floor(rowCount * selF0 * selF1));

  // Index hits and structured penalties
  const penalties: { name: string; value: number }[] = [];
  if (choice.hasTimeWindow && f1 && !f1.index) {
    penalties.push({ name: 'time-filter-not-indexed', value: 1.0 });
  }
  if (f0 && !f0.index) {
    penalties.push({ name: `eq-non-index:${f0.name}`, value: 0.3 });
  }

  const indexHits =
    (f0?.index ? 1 : 0) +
    (choice.hasTimeWindow && f1?.index ? 1 : 0);

  return { indexHits, estRows, penalties };
}