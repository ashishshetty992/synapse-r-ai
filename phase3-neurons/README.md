# Synapse R Phase 3 - Neurons Service

## Quick Start

Start the service:
```bash
uvicorn app.main:app --reload --port 5050
```

## Auto-Question Generator (AIQL)

Generate 5 safe example queries per entity (no `id`/`timestamp` pivots), write coverage, and validate:

```bash
# From repo root:
npm run regen:aiql
# or via make:
make regen-aiql

# From phase3-neurons directory:
curl -s -X POST localhost:5050/generate \
  -H 'x-api-key: SYNAPSE@2025' \
  -H 'content-type: application/json' \
  -d '{"perEntity":5,"outDir":"examples/aiql","writeFiles":true,"writeCoverage":true,"strict":true}' | jq
```

Outputs:
- `examples/aiql/*.iql.json` – AIQL example files
- `examples/aiql/_coverage.json` – per-entity role coverage

To adjust behavior:
- **Strict mode** (default): drops unsafe pivots. Set `"strict": false` in the /generate request to emit all, then validator will fail unsafe files (useful for debugging).
- **perEntity**: change how many examples per entity.

## API Endpoints

### IEM Management
- `POST /iem/build` - Build IEM from UEM
- `GET /iem/show` - Show IEM schema

### Intent Encoding
- `POST /intent/encode_nl` - Encode natural language to intent vector

### Synapse Matching
- `POST /synapse/match` - Match fields/entities against intent
- `POST /synapse/fill` - Slot filling + conflict resolution + AlignPlus™
- `POST /synapse/paths` - Join path inference with PathScore™
- `POST /synapse/explain` - Explain slot filling decisions

### Generation
- `POST /generate` - Generate AIQL examples

## Configuration

- `config/global/shaping.json` - Shaping weights and formulas
- `config/global/entity.json` - Entity embedding config
- `config/global/role.json` - Role definitions
- `synonyms.json` - Synonym mappings
- `config/global/time.json` - Time grammar config
