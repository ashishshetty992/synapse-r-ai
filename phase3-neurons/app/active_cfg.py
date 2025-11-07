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

def active_principles_path() -> str | None:
    sp = active_shaping_path()
    if not sp:
        return None
    root = os.path.dirname(sp)
    cand = os.path.join(root, "principles.json")
    return cand if os.path.exists(cand) else None

def load_json_safe(p):
    if not p or not os.path.exists(p):
        return None
    with open(p, "r") as f:
        return json.load(f)

def load_active_config_merged():
    """Return shaping config merged with principles → pathScoring/fill toggles."""
    sp = active_shaping_path()
    if not sp:
        # No active checkpoint, return empty dict (will use defaults in synapse.py)
        return {}
    
    shaping = load_json_safe(sp) or {}
    principles = load_json_safe(active_principles_path()) or {}
    cfg = dict(shaping)  # shallow copy is fine for our keys
    ps = cfg.setdefault("pathScoring", {})
    fill = cfg.setdefault("fill", {})
    enabled = (principles.get("enabled") or {}) if isinstance(principles.get("enabled"), dict) else {}
    
    # map toggles
    if enabled.get("hubPenalty") is True:
        ps["hubPenalty"] = ps.get("hubPenalty", 0.12)  # sensible default if absent
        ps["hubDegreeThreshold"] = ps.get("hubDegreeThreshold", 8)
    if enabled.get("unknownBackoff") is True:
        fill["unknownBackoff"] = True
    
    # keep a breadcrumb
    cfg.setdefault("_merged", {})["principlesEnabled"] = list(k for k, v in enabled.items() if v is True)
    return cfg

