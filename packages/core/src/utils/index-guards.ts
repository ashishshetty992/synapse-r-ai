import type { UEM } from '../types';

const RANGE_OPS = new Set(['gt','gte','lt','lte','between']);

export function requireIndexed(uem: UEM, entity: string, where: any): void {
  const ent = uem.entities.find((e) => e.name === entity);
  if (!ent) return;

  const indexed = new Set(ent.fields.filter((f) => f.index).map((f) => f.name));

  const check = (w: any): void => {
    if (!w) return;
    if ('and' in w && Array.isArray(w.and)) return w.and.forEach(check);
    if ('or'  in w && Array.isArray(w.or))  return w.or.forEach(check);

    if (RANGE_OPS.has(w.op)) {
      if (!indexed.has(w.field)) {
        const err: any = new Error(`Range on non-indexed field: ${w.field}`);
        err.code = 'NQL_INDEX_REQUIRED';
        throw err;
      }
    }
  };

  check(where);
}