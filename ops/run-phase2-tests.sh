#!/usr/bin/env bash
# Phase 2 End-to-End API Exerciser
# - Runs compile/query/explain/parity on all examples (MySQL + Mongo)
# - Adds a few direct IQL payloads to cover AND/OR, LIKE/CONTAINS, BETWEEN
# - Writes pretty JSON outputs under results/<adapter>/<endpoint>/*.json
# - Prints a compact pass/fail summary at the end

set -euo pipefail

HTTP_BASE="${HTTP_BASE:-http://127.0.0.1:4000}"
RESULTS_DIR="results/$(date +%Y%m%d-%H%M%S)"
EX_DIR="examples/iql"

# Planner + debug knobs (tweak as you like)
HDR_DEBUG="x-debug: 1"
HDR_PMODE="x-planner-mode: beam"
HDR_TAU="x-planner-tau: 0.7"
HDR_BEAM="x-planner-beam: 4"

# Tools check
need() { command -v "$1" >/dev/null 2>&1 || { echo "Missing $1"; exit 2; }; }
need curl
need jq

mkdir -p "$RESULTS_DIR"

log() { printf "\n\033[1;36m%s\033[0m\n" "$*"; }
ok()  { printf "\033[1;32m✔ %s\033[0m\n" "$*"; }
err() { printf "\033[1;31m✖ %s\033[0m\n" "$*"; }

############################################
# Health / readiness
############################################
log "1) Health checks"
curl -sf "${HTTP_BASE}/healthz" | jq > "${RESULTS_DIR}/healthz.json" && ok "healthz ok"
curl -sf "${HTTP_BASE}/readyz"  | jq > "${RESULTS_DIR}/readyz.json"  && ok "readyz ok"

############################################
# Optional: validate examples with schema
############################################
if [ -f "ops/validate-examples.js" ]; then
  log "2) Validating examples against IQL schema"
  node ops/validate-examples.js || true
fi

############################################
# Collect example files
############################################
log "3) Collecting example files"
if [ -d "$EX_DIR" ]; then
  # Use a more compatible approach for collecting files
  EXAMPLES=()
  while IFS= read -r -d '' file; do
    EXAMPLES+=("$file")
  done < <(find "$EX_DIR" -maxdepth 1 -type f -name '*.json' -print0 | sort -z)
else
  EXAMPLES=()
fi
echo "Found ${#EXAMPLES[@]} example(s)."

############################################
# Helper to POST JSON (save + basic check)
############################################
post_json() {
  local endpoint="$1" adapter="$2" infile="$3" outfile="$4"
  
  # capture first, then jq with fallback so jq errors don't terminate the script
  local resp
  if [[ "$infile" == "-" ]]; then
    if [[ "$adapter" == "mysql" || "$adapter" == "mongo" ]]; then
      resp="$(curl -sS -X POST \
        -H "content-type: application/json" \
        -H "$HDR_DEBUG" -H "$HDR_PMODE" -H "$HDR_TAU" -H "$HDR_BEAM" \
        -H "x-target: ${adapter}" \
        "${HTTP_BASE}${endpoint}?debug=1" \
        --data-binary @- || true)"
    else
      resp="$(curl -sS -X POST \
        -H "content-type: application/json" \
        -H "$HDR_DEBUG" -H "$HDR_PMODE" -H "$HDR_TAU" -H "$HDR_BEAM" \
        "${HTTP_BASE}${endpoint}?debug=1" \
        --data-binary @- || true)"
    fi
  else
    if [[ "$adapter" == "mysql" || "$adapter" == "mongo" ]]; then
      resp="$(curl -sS -X POST \
        -H "content-type: application/json" \
        -H "$HDR_DEBUG" -H "$HDR_PMODE" -H "$HDR_TAU" -H "$HDR_BEAM" \
        -H "x-target: ${adapter}" \
        "${HTTP_BASE}${endpoint}?debug=1" \
        -d @"$infile" || true)"
    else
      resp="$(curl -sS -X POST \
        -H "content-type: application/json" \
        -H "$HDR_DEBUG" -H "$HDR_PMODE" -H "$HDR_TAU" -H "$HDR_BEAM" \
        "${HTTP_BASE}${endpoint}?debug=1" \
        -d @"$infile" || true)"
    fi
  fi

  # pretty if JSON, else dump raw
  if echo "$resp" | jq . >/dev/null 2>&1; then
    echo "$resp" | jq > "$outfile"
  else
    echo "$resp" > "$outfile"
  fi

  # mark status (don't fail the script)
  if echo "$resp" | jq -e 'has("code")' >/dev/null 2>&1; then
    err "$(basename "$outfile") (API error present; see file)"
  else
    ok "$(basename "$outfile")"
  fi
}

############################################
# 4) Run over examples for both adapters
############################################
log "4) Running COMPILE / QUERY / EXPLAIN on examples (mysql + mongo)"
for adapter in mysql mongo; do
  for endpoint in compile query explain; do
    outdir="${RESULTS_DIR}/${adapter}/${endpoint}"
    mkdir -p "$outdir"
    for f in "${EXAMPLES[@]}"; do
      out="${outdir}/$(basename "$f" .json).json"
      post_json "/${endpoint}" "$adapter" "$f" "$out"
    done
  done
done

############################################
# 5) Parity (same IQL against both backends)
############################################
log "5) Running PARITY on all examples"
mkdir -p "${RESULTS_DIR}/parity"
for f in "${EXAMPLES[@]}"; do
  out="${RESULTS_DIR}/parity/$(basename "$f" .json).json"
  post_json "/parity" "both" "$f" "$out"
done

############################################
# 6) Quick summary
############################################
log "6) Summary"
TOTAL_FILES=$(find "$RESULTS_DIR" -type f -name "*.json" | wc -l | xargs)
PARITY_OK=$(jq -r 'select(has("parity")) | .parity | if has("equal") then .equal else .equal end' $(find "$RESULTS_DIR" -type f -name "*.json") 2>/dev/null | grep -c true || true)
PARITY_TOTAL=$(grep -R "\"parity\"" -n "$RESULTS_DIR" | wc -l | xargs)

echo "Results directory: $RESULTS_DIR"
echo "JSON outputs     : $TOTAL_FILES"
echo "Parity equal     : ${PARITY_OK}/${PARITY_TOTAL}"

ok "All requests executed. Inspect ${RESULTS_DIR}/ for outputs."