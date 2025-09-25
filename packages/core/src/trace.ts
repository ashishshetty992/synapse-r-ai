// packages/core/src/trace.ts
// ProofTrace (V2) types

export interface CostPenalty {
  name: string;
  value: number;
}

export interface CostSketch {
  indexHits: number;
  estRows: number;
  penalties: CostPenalty[];
}

export interface SlotCandidate {
  slot: string;                   // e.g. 'target', 'filter.field[1]'
  value: string;                  // chosen field
  score: number;                  // alignment score
  breakdown?: { align: number; cost: number; parsimony: number };
  source?: 'iem' | 'heuristic';
}

// ---- additive flags/timings envelope (no breaking changes) ----
export interface TraceFlags {
  shortCircuit?: boolean; // planner stopped early (e.g., qualified target path)
}

export interface ProofTraceV2 {
  alignments?: Array<{ intent: string; schema: string; score: number }>;
  slotCandidates?: SlotCandidate[];
  joins: Array<{
    from?: string; to?: string; via?: string;
    fanout?: number;
    alt?: Array<{ path: string; score: number }>;
  }>;
  cost: CostSketch;
  policy?: { tenancy?: any; rbac?: any };
  planner?: { mode: 'deterministic' | 'beam'; tau?: number; beamWidth?: number };

  // ---- new (additive) ----
  flags?: TraceFlags;     // e.g., { shortCircuit: true }
  planMs?: number;        // ms spent in planning (HTTP measured)
  adapterMs?: number;     // ms spent in adapter.read / adapter.explain
  rowCount?: number;      // number of rows returned (for /query)
  errorCode?: string;     // if failed, mapped taxonomy code (VALIDATION/TRANSLATION/ADAPTER/DB/...)
}