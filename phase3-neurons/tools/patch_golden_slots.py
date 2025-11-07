#!/usr/bin/env python3

import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "golden" / "acme"

def wants_timestamp_slot(g):
    try:
        intent = g.get("intent", {})
        return (intent.get("ask") == "top_k" and
                str(intent.get("target","")).endswith("orders.shipping_city"))
    except Exception:
        return False

def main():
    changed = 0
    for p in GOLDEN.glob("*.json"):
        try:
            g = json.loads(p.read_text())
            if wants_timestamp_slot(g):
                exp = g.setdefault("expected", {})
                slots = exp.setdefault("slots", {})
                if "timestamp" not in slots:
                    slots["timestamp"] = "orders.created_at"
                    p.write_text(json.dumps(g, indent=2))
                    changed += 1
                    print(f"Patched: {p.name}")
        except Exception as e:
            print(f"Error processing {p.name}: {e}")
    print(f"Patched {changed} golden files.")

if __name__ == "__main__":
    main()

