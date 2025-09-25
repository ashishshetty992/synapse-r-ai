// --------------------
// Scalar & schema
// --------------------
export type Scalar =
  | 'string'
  | 'number'
  | 'boolean'
  | 'datetime'
  | 'json';

export type FieldRole =
  | 'id'
  | 'timestamp'
  | 'quantity'
  | 'money'
  | 'geo'
  | 'category'
  | 'text'
  | 'unknown';

export interface Field {
  name: string;
  type: Scalar | { arrayOf: Scalar };
  required?: boolean;
  unique?: boolean;
  index?: boolean;
  pii?: boolean;
  role?: FieldRole;
  ref?: { entity: string; field: string };
  ndv?: number; // num distinct values (for rough selectivity)
}

export interface Entity {
  name: string;
  primaryKey: string;
  fields: Field[];
  softDelete?: boolean;
  rowCount?: number;
}

export interface UEM {
  version: string;
  entities: Entity[];
}

// --------------------
// Where / operators
// --------------------
// Unify the operator set used by both IQL->UQL and adapters.
export type WhereOp =
  | 'eq' | 'neq'
  | 'gt' | 'gte' | 'lt' | 'lte'
  | 'in' | 'nin'
  | 'between'
  | 'exists'
  | 'is'        // e.g. 'null' | 'not_null' | 'true' | 'false' (adapter-specific)
  | 'like'      // prefix/suffix/substring depends on adapter policy
  | 'contains'; // generic substring (Mongo regex or similar)

// Single Where definition
export type Where =
  | { and: Where[] }
  | { or:  Where[] }
  | { field: string; op: WhereOp; value?: any };

// --------------------
// Joins (1-hop)
// --------------------
export type JoinSpec = {
  type: 'inner' | 'left';
  from: string; // base entity/table
  to: string;   // joined entity/table
  on: { left: string; right: string }; // join keys (column names, unqualified)
};

// --------------------
// ReadQuery (UQL runtime form for adapters)
// --------------------
export interface ReadQuery {
  entity: string;
  joins?: JoinSpec[]; // optional joins
  select: { expr: string; as?: string }[];
  where?: Where;
  groupBy?: string[];
  orderBy?: { field: string; dir: 'asc' | 'desc' }[];
  limit?: number;
  offset?: number;
  cursor?: string | null;
  hints?: Record<string, any>;
  version: string;
}

// --------------------
// Adapter
// --------------------
export interface Adapter {
  name: 'mysql' | 'mongodb';
  init(uem: UEM, cfg: Record<string, any>): Promise<void>;
  read(q: ReadQuery): Promise<{ rows: any[]; meta: any }>;
  explain(q: ReadQuery): Promise<any>;
  health(): Promise<{ ok: boolean; details?: any }>;
}