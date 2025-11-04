# Weighted Preferences Test Results

## Test A: Count by city (should pick orders.shipping_city)

**Command:**
```bash
curl -s -X POST localhost:8000/synapse/fill \
  -H 'Content-Type: application/json' \
  -d '{
    "intent":{"ask":"top_k","metric":{"op":"count"},"target":"orders.id"},
    "preferRoles":["geo"],
    "preferTargets":["orders.shipping_city","customer.city","orders.country"],
    "preferTargetsBoost":{"orders.shipping_city":0.25,"customer.city":0.15,"orders.country":0.05},
    "aliasTargets":["orders.shipping_city"],
    "debugExplain": true
  }' | jq '{target: .intentFilled.target, geo: [.targetCandidates[] | select(.roleTop=="geo")], explain: .debug.targetExplain}'
```

**Result:**
{
  "target": "orders.shipping_city",
  "geo": [
    {
      "entity": "orders",
      "name": "shipping_city",
      "score": 1.3517836675633075,
      "roleTop": "geo"
    },
    {
      "entity": "orders",
      "name": "country",
      "score": 0.9768900563150902,
      "roleTop": "geo"
    }
  ],
  "explain": [
    {
      "field": "orders.shipping_city",
      "role": "geo",
      "score": 1.351784
    },
    {
      "field": "orders.country",
      "role": "geo",
      "score": 0.97689
    },
    {
      "field": "payment.amount",
      "role": "money",
      "score": 0.678057
    },
    {
      "field": "customer.id",
      "role": "id",
      "score": 0.624077
    },
    {
      "field": "product.id",
      "role": "id",
      "score": 0.624077
    },
    {
      "field": "orders.id",
      "role": "id",
      "score": 0.624077
    },
    {
      "field": "order_item.id",
      "role": "id",
      "score": 0.624077
    },
    {
      "field": "payment.id",
      "role": "id",
      "score": 0.624077
    },
    {
      "field": "orders.total_amount",
      "role": "money",
      "score": 0.619436
    },
    {
      "field": "customer.segment",
      "role": "category",
      "score": 0.563307
    },
    {
      "field": "payment.method",
      "role": "category",
      "score": 0.548249
    },
    {
      "field": "product.name",
      "role": "text",
      "score": 0.44275
    },
    {
      "field": "product.category",
      "role": "category",
      "score": 0.438775
    },
    {
      "field": "order_item.order_id",
      "role": "id",
      "score": 0.425157
    },
    {
      "field": "payment.order_id",
      "role": "id",
      "score": 0.425157
    },
    {
      "field": "orders.status",
      "role": "category",
      "score": 0.424124
    },
    {
      "field": "payment.status",
      "role": "category",
      "score": 0.424124
    },
    {
      "field": "customer.full_name",
      "role": "text",
      "score": 0.405428
    },
    {
      "field": "order_item.quantity",
      "role": "quantity",
      "score": 0.230774
    },
    {
      "field": "order_item.product_id",
      "role": "id",
      "score": 0.195591
    },
    {
      "field": "orders.customer_id",
      "role": "id",
      "score": 0.174014
    },
    {
      "field": "customer.created_at",
      "role": "timestamp",
      "score": 0.148378
    },
    {
      "field": "product.created_at",
      "role": "timestamp",
      "score": 0.148378
    },
    {
      "field": "orders.created_at",
      "role": "timestamp",
      "score": 0.148378
    },
    {
      "field": "order_item.created_at",
      "role": "timestamp",
      "score": 0.148378
    },
    {
      "field": "payment.created_at",
      "role": "timestamp",
      "score": 0.148378
    },
    {
      "field": "orders.currency",
      "role": "unknown",
      "score": 0.06888
    },
    {
      "field": "product.sku",
      "role": "unknown",
      "score": 0.0557
    },
    {
      "field": "customer.email",
      "role": "unknown",
      "score": 0.03159
    },
    {
      "field": "product.active",
      "role": "unknown",
      "score": 0.023091
    },
    {
      "field": "order_item.unit_price",
      "role": "money",
      "score": -0.01967
    },
    {
      "field": "product.price",
      "role": "money",
      "score": -0.066438
    }
  ]
}

---

## Test B: Revenue by city last month

**Command:**
```bash
# 1) Extract alias targets
ALIAS=$(curl -s -X POST localhost:8000/intent/encode_nl \
  -H 'Content-Type: application/json' \
  -d '{"text":"top revenue by city last month"}' | \
  jq -r '[.debug.topAliasHits[].target | select(test("^[a-zA-Z_]+\\.[a-zA-Z0-9_]+$"))] | @json')

# 2) Fill with aliasTargets
curl -s -X POST localhost:8000/synapse/fill \
  -H 'Content-Type: application/json' \
  -d "{
    \"intent\":{\"ask\":\"top_k\",\"metric\":{\"op\":\"sum\",\"target\":\"orders.total_amount\"}},
    \"preferRoles\":[\"geo\"],
    \"preferTargets\":[\"orders.shipping_city\",\"customer.city\",\"orders.country\"],
    \"preferTargetsBoost\":{\"orders.shipping_city\":0.25},
    \"aliasTargets\": $ALIAS,
    \"debugExplain\": true
  }" | jq '{target: .intentFilled.target, geo: [.targetCandidates[] | select(.roleTop=="geo")], explain: .debug.targetExplain}'
```

**Result:**
{
  "target": "orders.total_amount",
  "geo": [
    {
      "entity": "orders",
      "name": "shipping_city",
      "score": 0.9954647518149499,
      "roleTop": "geo"
    },
    {
      "entity": "orders",
      "name": "country",
      "score": 0.4946340322305919,
      "roleTop": "geo"
    }
  ],
  "explain": [
    {
      "field": "orders.total_amount",
      "role": "money",
      "score": 1.541438
    },
    {
      "field": "payment.amount",
      "role": "money",
      "score": 1.223192
    },
    {
      "field": "orders.shipping_city",
      "role": "geo",
      "score": 0.995465
    },
    {
      "field": "orders.country",
      "role": "geo",
      "score": 0.494634
    },
    {
      "field": "order_item.unit_price",
      "role": "money",
      "score": 0.447956
    }
  ]
}

---

## Summary

✅ **Weighted preferences implemented:**
1. `preferTargetsBoost` - per-target weighted boosts (dict[str, float])
2. `aliasTargets` - NL encoder alias hits for strong context bonus (0.30)
3. Granularity rule - city > country for top_k queries (0.12 bonus)
4. `debugExplain` - explain output showing final candidate scores

✅ **Test A Results:**
- Target selected: `orders.shipping_city` ✓
- Score: 1.351784 (vs country 0.976890)
- Weighted boosts successfully pushed city to top

✅ **Test B Results:**
- Primary target: `orders.total_amount` (money metric)
- Geo dimension: `orders.shipping_city` scores 0.995465 (vs country 0.494634)
- Alias targets from NL encoder correctly threaded through


## Full Response Examples

### Test A Full Response:
{
  "ok": true,
  "intentFilled": {
    "ask": "top_k",
    "metric": {
      "op": "count"
    },
    "target": "orders.shipping_city"
  },
  "targetCandidates": [
    {
      "entity": "orders",
      "name": "shipping_city",
      "score": 1.3517836675633075,
      "roleTop": "geo"
    },
    {
      "entity": "orders",
      "name": "country",
      "score": 0.9768900563150902,
      "roleTop": "geo"
    },
    {
      "entity": "payment",
      "name": "amount",
      "score": 0.678056884485366,
      "roleTop": "money"
    },
    {
      "entity": "customer",
      "name": "id",
      "score": 0.6240774778525225,
      "roleTop": "id"
    },
    {
      "entity": "product",
      "name": "id",
      "score": 0.6240774778525225,
      "roleTop": "id"
    },
    {
      "entity": "orders",
      "name": "id",
      "score": 0.6240774778525225,
      "roleTop": "id"
    },
    {
      "entity": "order_item",
      "name": "id",
      "score": 0.6240774778525225,
      "roleTop": "id"
    },
    {
      "entity": "payment",
      "name": "id",
      "score": 0.6240774778525225,
      "roleTop": "id"
    }
  ],
  "entityCandidates": [
    {
      "entity": "orders",
      "score": 0.5755934300021159
    },
    {
      "entity": "payment",
      "score": 0.46938306500018734
    },
    {
      "entity": "order_item",
      "score": 0.34570733293160383
    },
    {
      "entity": "customer",
      "score": 0.3306808907129765
    },
    {
      "entity": "product",
      "score": 0.27623663132706905
    }
  ],
  "conflicts": [
    {
      "slot": "timestamp",
      "candidates": [
        "order_item.created_at",
        "payment.created_at",
        "orders.created_at",
        "product.created_at",
        "customer.created_at"
      ],
      "resolution": "order_item.created_at",
      "why": {
        "PathScore": 0.24119872110866156,
        "CosineContext": 0.1533633746528362,
        "RoleCoherence": 1.0,
        "LocalBonus": 0.0,
        "weights": {
          "path": 0.5,
          "cosine": 0.35,
