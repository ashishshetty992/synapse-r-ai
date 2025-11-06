import os, json, logging
from functools import lru_cache
from .active_cfg import active_shaping_path

LOG = logging.getLogger(__name__)

try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
        if os.environ.get("NEURONS_DEBUG_BOOT", "0") == "1":
            print(f"DEBUG: Loaded .env from {env_path}")
    else:
        load_dotenv()
        if os.environ.get("NEURONS_DEBUG_BOOT", "0") == "1":
            print(f"DEBUG: Tried to load .env from current directory")
except ImportError:
    if os.environ.get("NEURONS_DEBUG_BOOT", "0") == "1":
        print("DEBUG: python-dotenv not installed, skipping .env loading")
except Exception as e:
    if os.environ.get("NEURONS_DEBUG_BOOT", "0") == "1":
        print(f"DEBUG: Failed to load .env: {e}")

if os.environ.get("NEURONS_DEBUG_BOOT", "0") == "1":
    print("\n=== Environment Variables (NEURONS_*) ===")
    neurons_vars = {k: v for k, v in os.environ.items() if k.startswith("NEURONS_") or k.startswith("TRAINER_")}
    if neurons_vars:
        for k, v in sorted(neurons_vars.items()):
            print(f"{k}={v}")
    else:
        print("No NEURONS_* or TRAINER_* environment variables found")
    print("==========================================\n")

CFG_ROOT = os.environ.get("NEURONS_CFG_ROOT", os.path.join(os.getcwd(), "config", "tenant"))
if CFG_ROOT.endswith("tenant") or CFG_ROOT.endswith("tenant/"):
    GLOBAL_DIR = os.path.join(os.path.dirname(CFG_ROOT.rstrip("/")), "global")
else:
    GLOBAL_DIR = os.path.join(CFG_ROOT, "global")
SYNONYMS_PATH = os.environ.get("NEURONS_SYNONYMS", os.path.join(os.getcwd(), "synonyms.json"))
CFG_HOT = os.environ.get("NEURONS_CFG_HOT", "0") == "1"

def _load_json(path):
    if not path or not os.path.exists(path): return None
    with open(path, "r") as f: return json.load(f)

def _mtime(path: str) -> float:
    try: return os.path.getmtime(path)
    except Exception: return 0.0

@lru_cache(maxsize=256)
def _load_role_cfg_cached(tenant: str | None, t_mtime: float, g_mtime: float):
    tpath = os.path.join(CFG_ROOT, tenant, "role.json") if tenant else None
    gpath = os.path.join(GLOBAL_DIR, "role.json")
    t = _load_json(tpath) if tpath else None
    g = _load_json(gpath)
    LOG.debug("load_role_cfg_cached(tenant=%s) -> %s", tenant, (t or g or {"keywords": {}}))
    return (t or g or {"keywords": {}})

def load_role_cfg(tenant: str | None):
    tpath = os.path.join(CFG_ROOT, tenant, "role.json") if tenant else None
    gpath = os.path.join(GLOBAL_DIR, "role.json")
    LOG.debug("load_role_cfg(tenant=%s) -> %s", tenant, (tpath or gpath))
    return _load_role_cfg_cached(tenant, _mtime(tpath) if tpath else 0.0, _mtime(gpath))

@lru_cache(maxsize=256)
def _load_shaping_cfg_cached(tenant: str | None, t_mtime: float, g_mtime: float):
    tpath = os.path.join(CFG_ROOT, tenant, "shaping.json") if tenant else None
    gpath = os.path.join(GLOBAL_DIR, "shaping.json")
    t = _load_json(tpath) if tpath else None
    g = _load_json(gpath)
    result = t or g
    if not result:
        # Return proper defaults matching the structure
        return {"weights": {}, "pathScoring": {}, "alignPlus": {}}
    LOG.debug("load_shaping_cfg_cached(tenant=%s) -> %s", tenant, result)
    return result

def load_shaping_cfg(tenant: str | None, *, allow_active: bool = True):
    p_active = active_shaping_path()
    if allow_active and p_active:
        LOG.debug(f"load_shaping_cfg(active) -> {p_active}")
        with open(p_active) as f:
            return json.load(f)
    
    # use cached path for tenant/global
    tpath = os.path.join(CFG_ROOT, tenant, "shaping.json") if tenant else None
    gpath = os.path.join(GLOBAL_DIR, "shaping.json")
    return _load_shaping_cfg_cached(tenant, _mtime(tpath) if tpath else 0.0, _mtime(gpath))

@lru_cache(maxsize=256)
def _load_entity_cfg_cached(tenant: str | None, t_mtime: float, g_mtime: float):
    tpath = os.path.join(CFG_ROOT, tenant, "entity.json") if tenant else None
    gpath = os.path.join(GLOBAL_DIR, "entity.json")
    dflt = {
        "entity_blend": {"entityWeight": 0.70, "fieldWeight": 0.30, "entityAliasAlpha": 0.15},
        "centroid_gamma": 0.60,
        "role_alpha": 0.12,
    }
    t = _load_json(tpath) if tpath else None
    g = _load_json(gpath)
    LOG.debug("load_entity_cfg_cached(tenant=%s) -> %s", tenant, (t or g or dflt))
    return (t or g or dflt)

def load_entity_cfg(tenant: str | None):
    tpath = os.path.join(CFG_ROOT, tenant, "entity.json") if tenant else None
    gpath = os.path.join(GLOBAL_DIR, "entity.json")
    LOG.debug("load_entity_cfg(tenant=%s) -> %s", tenant, (tpath or gpath))
    return _load_entity_cfg_cached(tenant, _mtime(tpath) if tpath else 0.0, _mtime(gpath))

@lru_cache(maxsize=1)
def load_synonyms_cached():
    return _load_json(SYNONYMS_PATH) or {}

def load_synonyms():
    if CFG_HOT:
        return _load_json(SYNONYMS_PATH) or {}
    return load_synonyms_cached()

@lru_cache(maxsize=256)
def _load_time_cfg_cached(tenant: str | None, t_mtime: float, g_mtime: float):
    tpath = os.path.join(CFG_ROOT, tenant, "time.json") if tenant else None
    gpath = os.path.join(GLOBAL_DIR, "time.json")
    t = _load_json(tpath) if tpath else None
    g = _load_json(gpath) or {}

    dflt = {
        "relative": {},
        "aliases": {},
        "fiscal": {"enabled": False, "year_start_month": 1}
    }

    out = {
        "relative": dict(g.get("relative", {})),
        "aliases": dict(g.get("aliases", {})),
        "fiscal": dict(g.get("fiscal", {})) or dict(dflt["fiscal"]),
    }
    if isinstance(t, dict):
        if "relative" in t and isinstance(t["relative"], dict):
            out["relative"].update(t["relative"])
        if "aliases" in t and isinstance(t["aliases"], dict):
            out["aliases"].update(t["aliases"])
        if "fiscal" in t and isinstance(t["fiscal"], dict):
            out["fiscal"].update(t["fiscal"])

    out["relative"] = out.get("relative", {}) or {}
    out["aliases"]  = out.get("aliases",  {}) or {}
    if "fiscal" not in out: out["fiscal"] = dict(dflt["fiscal"])
    if "enabled" not in out["fiscal"]: out["fiscal"]["enabled"] = False
    if "year_start_month" not in out["fiscal"]: out["fiscal"]["year_start_month"] = 1

    LOG.debug("load_time_cfg_cached(tenant=%s) -> rel=%d, alias=%d, fiscal=%s", tenant, len(out['relative']), len(out['aliases']), out['fiscal'])
    return out

def load_time_cfg(tenant: str | None):
    tpath = os.path.join(CFG_ROOT, tenant, "time.json") if tenant else None
    gpath = os.path.join(GLOBAL_DIR, "time.json")
    LOG.debug("load_time_cfg(tenant=%s) -> %s", tenant, (tpath or gpath))
    return _load_time_cfg_cached(tenant, _mtime(tpath) if tpath else 0.0, _mtime(gpath))
