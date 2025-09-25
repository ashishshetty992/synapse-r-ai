// packages/core/src/schemas.ts
import { z } from 'zod';

// shared op set (unchanged)
export const WhereOpEnum = z.enum([
  'eq','neq',
  'gt','gte','lt','lte',
  'in','nin',
  'between','exists','is',
  'like','contains', 'ilike'
]);

// recursive UQL where (unchanged)
export const UQLWhereLeaf = z.object({
  field: z.string(),
  op: WhereOpEnum,
  value: z.any().optional()
}).strict();
export const UQLWhere: any = z.lazy(() =>
  z.union([
    z.object({ and: z.array(UQLWhere) }).strict(),
    z.object({ or: z.array(UQLWhere) }).strict(),
    UQLWhereLeaf
  ])
);

// IQL filter (incoming) (unchanged)
export const IQLFilter = z.object({
  field: z.string(),
  op: WhereOpEnum,
  value: z.any().optional()
}).strict();

// ---- Additive helper schemas for new asks ----
const ComparePeriod = z.object({
  label: z.string(),
  start: z.string(), // keep as string to avoid breaking callers; adapters can validate datetime
  end: z.string()
}).strict();

const IQLSelectExpr = z.object({
  expr: z.string(),
  as: z.string().optional()
}).strict();

// IQL schema (additive only)
export const IQLSchema = z.object({
  ask: z.enum(['top_k','trend','compare','detail']),

  // trend
  grain: z.enum(['day','week','month']).optional(),

  // top_k
  k: z.number().int().positive().optional(),
  target: z.string().optional(),

  // metric (add count_distinct)
  metric: z.object({
    name: z.string().optional(),
    op: z.enum(['count','sum','avg','min','max','count_distinct']).optional(),
    over: z.string().optional()
  }).optional(),

  // filters/time
  filters: z.array(IQLFilter).optional(),
  timeWindow: z.object({ start: z.string(), end: z.string() }).optional(),

  // ordering/paging (detail + general)
  orderBy: z.array(z.object({ field: z.string(), dir: z.enum(['asc','desc']) })).optional(),
  limit: z.number().int().positive().optional(),
  offset: z.number().int().min(0).optional(),   // <-- added offset for paging
  cursor: z.string().nullable().optional(),

  // detail: allow explicit column selection in IQL (optional)
  select: z.array(IQLSelectExpr).optional(),

  // compare: multi-target and/or multi-period
  targets: z.array(z.string()).optional(),            // e.g., ['customer.segment','country']
  comparePeriods: z.array(ComparePeriod).optional(),  // e.g., [{label:'Aug',start:...,end:...}, {label:'Sep',...}]

  explain: z.boolean().optional(),
  version: z.string().optional()
}).strict();
export type IQL = z.infer<typeof IQLSchema>;

// UQL schema (+ joins) (mostly unchanged; additive-safe)
export const UQLJoin = z.object({
  type: z.enum(['inner','left']),
  from: z.string(),
  to: z.string(),
  on: z.object({ left: z.string(), right: z.string() }).strict()
}).strict();

export const UQLSchema = z.object({
  entity: z.string(),
  joins: z.array(UQLJoin).optional(),
  select: z.array(z.object({ expr: z.string(), as: z.string().optional() })).min(1),
  where: UQLWhere.optional(),
  groupBy: z.array(z.string()).optional(),
  orderBy: z.array(z.object({ field: z.string(), dir: z.enum(['asc','desc']) })).optional(),
  limit: z.number().int().positive().optional(),
  offset: z.number().int().min(0).optional(),       // <-- added to mirror IQL paging when planner passes through
  cursor: z.string().nullable().optional(),
  hints: z.object({ requireIndexedFilters: z.boolean().optional() }).passthrough().optional(),
  version: z.literal('uql/0.1')
}).strict();
export type UQL = z.infer<typeof UQLSchema>;

// Errors (unchanged)
export const Errors = {
  IQL_VALIDATION: (msg:string) => ({ code: 'NQL_IQL_VALIDATION', message: msg }),
  ENTITY_AMBIGUOUS: (alts:string[]) => ({ code: 'NQL_ENTITY_AMBIGUOUS', message: 'Multiple plausible entities', details: { candidates: alts } }),
  FIELD_UNRESOLVED: (name:string) => ({ code: 'NQL_FIELD_UNRESOLVED', message: `Cannot resolve field: ${name}` }),
  INDEX_REQUIRED: () => ({ code: 'NQL_INDEX_REQUIRED', message: 'Range predicate requires an index (requireIndexedFilters=true)' })
} as const;