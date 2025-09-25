// v0 IEM: lightweight semantic layer inferred from UEM
export type Role = 'id'|'timestamp'|'quantity'|'money'|'geo'|'category'|'text'|'unknown';

export interface RoleProbs {
  id?: number; timestamp?: number; quantity?: number; money?: number;
  geo?: number; category?: number; text?: number; unknown?: number;
}

export interface JoinPrior {
  toEntity: string;   // e.g., "customer"
  via?: string;       // e.g., "customer_id"
  score: number;      // 0..1 confidence
}

export interface IEMFieldEmb {
  entity: string;     // e.g., "sales_order"
  name: string;       // e.g., "shipping_city"
  aliases: string[];  // ["shipping city", "city", ...]
  role: RoleProbs;    // role probabilities
  vec?: number[];     // unit-normalized embedding (same dim for all)
  joinPrior?: JoinPrior[];
}

export interface IEMEntityEmb {
  entity: string;
  vec?: number[];
}

export interface IEM {
  version: 'iem/0.1';
  fields: IEMFieldEmb[];
  entities?: IEMEntityEmb[];
}