#!/bin/bash
# Test script for /synapse/paths endpoint with debug features
# Run with: bash test_paths_debug.sh

BASE_URL="http://localhost:8000"
API_KEY="SYNAPSE@2025"
TENANT="acme"

echo "=== Testing /synapse/paths endpoint with debug features ==="
echo ""

# jq parity checker function
PARITY_CHECKER='def abs(x): if x<0 then -x else x end;
  . as $r
  | [ range(0; ($r.paths|length)) 
      | { i: .,
          score: ($r.paths[.].score // 0),
          final: ($r.debug.scoreComponentsByPath[.].final // 0),
          diff:  (($r.paths[.].score // 0) - ($r.debug.scoreComponentsByPath[.].final // 0))
        }
    ] as $rows
  | { ok: ( $rows | map( ( .diff | abs ) < 1e-6 ) | all ),
      rows: $rows
    }'

echo "=== A) Product category (2 paths) ==="
curl -s -X POST "${BASE_URL}/synapse/paths?features=1" \
  -H 'Content-Type: application/json' \
  -H "x-api-key: ${API_KEY}" \
  -H "x-tenant-id: ${TENANT}" \
  -d '{"intent":{"ask":"top_k","metric":{"op":"sum","target":"orders.total_amount"},"target":"product.category"},"topK":2}' \
  | tee /tmp/paths_cat.json \
  | jq '.debug.edgeScoresByPath[0], .debug.scoreComponentsByPath[0]'

echo ""
echo "Parity check:"
cat /tmp/paths_cat.json | jq "$PARITY_CHECKER"

echo ""
echo "=== B) Product category (3 paths) - scoreMax + normalized scores ==="
curl -s -X POST "${BASE_URL}/synapse/paths?features=1" \
  -H 'Content-Type: application/json' \
  -H "x-api-key: ${API_KEY}" \
  -H "x-tenant-id: ${TENANT}" \
  -d '{"intent":{"ask":"top_k","metric":{"op":"sum","target":"orders.total_amount"},"target":"product.category"},"topK":3}' \
  | jq '.debug.scoreMax, .debug.scoreNormTop'

echo ""
echo "=== C) Zero-hop case (goal = start entity) ==="
curl -s -X POST "${BASE_URL}/synapse/paths?features=1" \
  -H 'Content-Type: application/json' \
  -H "x-api-key: ${API_KEY}" \
  -H "x-tenant-id: ${TENANT}" \
  -d '{"intent":{"ask":"top_k","metric":{"op":"sum","target":"orders.total_amount"},"target":"orders.shipping_city"},"topK":2}' \
  | jq '.paths[0], .debug.edgeScoresByPath[0], .debug.scoreComponentsByPath[0]'

echo ""
echo "=== D) One-hop to customer.city ==="
curl -s -X POST "${BASE_URL}/synapse/paths?features=1" \
  -H 'Content-Type: application/json' \
  -H "x-api-key: ${API_KEY}" \
  -H "x-tenant-id: ${TENANT}" \
  -d '{"intent":{"ask":"top_k","metric":{"op":"sum","target":"orders.total_amount"},"target":"customer.city"},"topK":2}' \
  | tee /tmp/paths_city.json \
  | jq '.paths[0], .debug.edgeScoresByPath[0], .debug.scoreComponentsByPath[0]'

echo ""
echo "Parity check:"
cat /tmp/paths_city.json | jq "$PARITY_CHECKER"

echo ""
echo "=== E) Test knob overrides ==="
echo "Setting lighter priors..."
curl -s -X POST "${BASE_URL}/config/knobs" \
  -H 'Content-Type: application/json' \
  -H "x-api-key: ${API_KEY}" \
  -d '{"pathScoring":{"idPrior":0.08,"fkBonus":0.03,"lengthPriorBase":0.92}}' \
  | jq '.now.pathScoring'

echo ""
echo "Re-running category paths with new knobs..."
curl -s -X POST "${BASE_URL}/synapse/paths?features=1" \
  -H 'Content-Type: application/json' \
  -H "x-api-key: ${API_KEY}" \
  -H "x-tenant-id: ${TENANT}" \
  -d '{"intent":{"ask":"top_k","metric":{"op":"sum","target":"orders.total_amount"},"target":"product.category"},"topK":2}' \
  | jq '.debug.pathScoringUsed, .debug.edgeScoresByPath[0], .debug.scoreComponentsByPath[0]'

echo ""
echo "=== Tests complete ==="

