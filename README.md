# SYNAPSE-R Phase 1 — TS Runtime (Monolith)

**Goal:** Deliver a portable runtime that can take **IQL (intent)**, compile it to **UQL (execution IR)**, run it on **MySQL or Mongo**, and return results **with a Proof Trace**. No client configs. Our graph is **in-memory** (no Neo4j), keeping us effort-agnostic.

**USP:** “Don’t give me anything — I’ll do everything.”

## What’s here
- NQL v0.1 (IQL + UQL) with Zod validation
- UEM builder (introspect MySQL + Mongo) → `uem.json`
- Our **SchemaGraph** (custom, in-memory) for entities/fields/relations/indexes
- Planner v0 (heuristic, deterministic; τ exists but fixed)
- Adapters for MySQL (Kysely) + Mongo (native) with index-safety checks
- HTTP API with `/compile`, `/query`, `/explain`, `/healthz`
- Conformance tests skeleton

## Quickstart
```bash
pnpm i
# Boot infra (MySQL 8 + Mongo 7 + UIs)
docker compose -f ops/docker-compose.yml up -d
# Build everything
pnpm -r build
# Introspect demo DBs → builds uem.json
pnpm --filter @nauvra/uem introspect:demo
# Start API
pnpm --filter @nauvra/http dev

# Test
pnpm test


# Try a query (top 10 cities last month, India)
curl -s localhost:4000/query \
  -H 'content-type: application/json' \
  -H 'x-target: mysql' \
  -d @examples/iql/top-cities.last-month.in.json | jq
```

## API
- `POST /compile` → `{ uql, trace, warnings? }`
- `POST /query` → `{ rows, meta, uql, trace }` (use header `x-target: mysql|mongo`)
- `POST /explain` → adapter explain (SQL or Mongo pipeline) + trace
- `GET  /healthz` → health

---

## Recipes

Below are minimal IQL requests you can run via `curl`.  
Replace `x-target: mysql` with `mongo` to test Mongo backend.

### 1. `top_k` — top N groups by metric
```bash
curl -s localhost:4000/query \
  -H 'content-type: application/json' \
  -H 'x-target: mysql' \
  -d '{
    "ask": "top_k",
    "target": "shipping_city",
    "metric": { "op": "count" },
    "timeWindow": { "start": "2025-08-01", "end": "2025-08-31" },
    "k": 5
  }' | jq
```

### 2. `trend` — time series over a period
```bash
curl -s localhost:4000/query \
  -H 'content-type: application/json' \
  -H 'x-target: mysql' \
  -d '{
    "ask": "trend",
    "grain": "week",
    "metric": { "op": "sum", "over": "total_amount" },
    "timeWindow": { "start": "2025-08-01", "end": "2025-08-31" }
  }' | jq
```

### 3. `compare` (targets) — multi-group breakdown
```bash
curl -s localhost:4000/query \
  -H 'content-type: application/json' \
  -H 'x-target: mysql' \
  -d '{
    "ask": "compare",
    "targets": ["customer.segment", "country"],
    "metric": { "op": "count" },
    "timeWindow": { "start": "2025-08-01", "end": "2025-08-31" }
  }' | jq
```

### 4. `compare` (periods) — Aug vs Sep
```bash
curl -s localhost:4000/query \
  -H 'content-type: application/json' \
  -H 'x-target: mysql' \
  -d '{
    "ask": "compare",
    "targets": ["shipping_city"],
    "metric": { "op": "sum", "over": "total_amount" },
    "comparePeriods": [
      { "label": "Aug", "start": "2025-08-01", "end": "2025-08-31" },
      { "label": "Sep", "start": "2025-09-01", "end": "2025-09-30" }
    ]
  }' | jq
```

### 5. `detail` — drilldown with paging
```bash
curl -s localhost:4000/query \
  -H 'content-type: application/json' \
  -H 'x-target: mysql' \
  -d '{
    "ask": "detail",
    "filters": [
      { "field": "country", "op": "eq", "value": "IN" }
    ],
    "timeWindow": { "start": "2025-08-01", "end": "2025-08-31" },
    "orderBy": [{ "field": "created_at", "dir": "desc" }],
    "limit": 10,
    "offset": 0
  }' | jq
```

---

## Phase-2 hooks
- Python **iem-scorer** microservice for embeddings & plan scoring (gRPC/HTTP).
- Cost model + τ-controlled exploration in planner.
