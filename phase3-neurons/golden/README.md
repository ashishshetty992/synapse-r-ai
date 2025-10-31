Put small hand-labeled intents and expected slots here for evaluator.
Schema:
{
  "intent": {"ask": "top_k", "metric": {"op": "sum"}},
  "expected": {"target": "orders.shipping_city", "slots": {"timestamp": "orders.created_at"}},
  "notes": "Top city by revenue"
}