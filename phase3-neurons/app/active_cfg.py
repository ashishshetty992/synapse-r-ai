import os, json

ROOT = os.path.dirname(os.path.dirname(__file__))
ACTIVE_PTR = os.path.join(ROOT, "runtime", "active.json")

def _read_active():
    try:
        with open(ACTIVE_PTR, "r") as f:
            return json.load(f)
    except Exception:
        return None

def active_shaping_path():
    a = _read_active()
    return a.get("shaping") if a and a.get("shaping") and os.path.exists(a["shaping"]) else None

def active_fewshot_path():
    a = _read_active()
    return a.get("fewshot") if a and a.get("fewshot") and os.path.exists(a["fewshot"]) else None

