# Environment Variables

The following environment variables are used by the Phase 3 Neurons service:

## Security & Rate Limiting
- `NEURONS_API_KEY` - If set, requires `x-api-key` header for authentication
- `NEURONS_RL_MAX` - Maximum requests per rate limit window (default: 60)
- `NEURONS_RL_WINDOW` - Rate limit window size in seconds (default: 60)

## Request Limits
- `NEURONS_MAX_BODY` - Maximum request body size in bytes (default: 2,097,152 = 2MB)

## Output & Configuration
- `NEURONS_OUT_BASE` - Base directory for generated files; `/generate` will refuse paths outside this directory
- `NEURONS_TRUST_PROXY` - Set to "1" to trust `X-Forwarded-For` headers for IP resolution
- `NEURONS_CFG_ROOT` - Root directory for configuration files (default: `config/`)
- `NEURONS_SYNONYMS` - Path to synonyms.json file
- `NEURONS_CFG_HOT` - Set to "1" to enable hot-reloading of configs

## Paths
- `UEM_PATH` - Path to UEM (Universal Entity Model) JSON file
- `IEM_PATH` - Path to IEM (Intelligent Embedding Model) JSON file

## Optional Features
- `NEURONS_ROLE_CFG` - Optional JSON file for role configuration
- `NEURONS_SHAPING_CFG` - Optional JSON file for shaping configuration

## Trainer & Canary
- `TRAINER_SHADOW_ONLY` - Set to "true" to enable shadow mode (never change decisions; only attach shadow comparisons)
- `TRAINER_TENANT_CANARY` - Comma-separated list of tenant IDs allowed to use trainer knobs (e.g., "acme,shipyaari"). If empty, all tenants can use trainer knobs.

## Prometheus Metrics
The service exposes Prometheus metrics at `/metrics` endpoint if `prometheus_client` is installed.

## Request Headers
- `x-experiment: train-shadow` - Per-request header to emit shadow debug comparisons even if `TRAINER_SHADOW_ONLY` is false
- `x-tenant-id` - Tenant identifier for multi-tenant configurations

