# app/config.py
import os, json, logging
from functools import lru_cache

from starlette.background import P

LOG = logging.getLogger(__name__)

CFG_ROOT = os.environ.get("NEURONS_CFG_ROOT", os.path.join(os.getcwd(), "config"))
GLOBAL_DIR = os.path.join(CFG_ROOT, "global")
SYNONYMS_PATH = os.environ.get("NEURONS_SYNONYMS", os.path.join(os.getcwd(), "synonyms.json"))
CFG_HOT = os.environ.get("NEURONS_CFG_HOT", "0") == "1"

def _load_json(path):
    if not path or not os.path.exists(path): return None
    with open(path, "r") as f: return json.load(f)

def _mtime(path: str) -> float:
    try: return os.path.getmtime(path)
    except Exception: return 0.0

# ---------- ROLE ----------
@lru_cache(maxsize=256)
def _load_role_cfg_cached(tenant: str | None, t_mtime: float, g_mtime: float):
    tpath = os.path.join(CFG_ROOT, tenant, "role.json") if tenant else None
    gpath = os.path.join(GLOBAL_DIR, "role.json")
    t = _load_json(tpath) if tpath else None
    g = _load_json(gpath)
    print(f"DEBUG: _load_role_cfg_cached(tenant={tenant}) -> {t or g or {'keywords': {}}}")
    return (t or g or {"keywords": {}})

def load_role_cfg(tenant: str | None):
    tpath = os.path.join(CFG_ROOT, tenant, "role.json") if tenant else None
    gpath = os.path.join(GLOBAL_DIR, "role.json")
    print(f"DEBUG: load_role_cfg(tenant={tenant}) -> {tpath or gpath}")
    return _load_role_cfg_cached(tenant, _mtime(tpath) if tpath else 0.0, _mtime(gpath))

# ---------- SHAPING ----------
@lru_cache(maxsize=256)
def _load_shaping_cfg_cached(tenant: str | None, t_mtime: float, g_mtime: float):
    tpath = os.path.join(CFG_ROOT, tenant, "shaping.json") if tenant else None
    gpath = os.path.join(GLOBAL_DIR, "shaping.json")
    t = _load_json(tpath) if tpath else None
    g = _load_json(gpath)
    print(f"DEBUG: _load_shaping_cfg_cached(tenant={tenant}) -> {t or g or {'weights': {}}}")
    return (t or g or {"weights": {}})

def load_shaping_cfg(tenant: str | None):
    from .active_cfg import active_shaping_path
    
    # 1) prefer active checkpoint (global for now; tenant-level later)
    p_active = active_shaping_path()
    if p_active:
        LOG.debug(f"load_shaping_cfg(active) -> {p_active}")
        with open(p_active) as f:
            return json.load(f)
    
    # 2) fallback to tenant or global config
    if tenant:
        p = os.path.join(CFG_ROOT, tenant, "shaping.json")
        if os.path.exists(p):
            LOG.debug(f"load_shaping_cfg(tenant={tenant}) -> {p}")
            with open(p) as f:
                return json.load(f)
    
    p = os.path.join(GLOBAL_DIR, "shaping.json")
    LOG.debug(f"load_shaping_cfg(tenant={tenant}) -> {p}")
    with open(p) as f:
        return json.load(f)

# ---------- ENTITY ----------
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
    print(f"DEBUG: _load_entity_cfg_cached(tenant={tenant}) -> {t or g or dflt}")
    return (t or g or dflt)

def load_entity_cfg(tenant: str | None):
    tpath = os.path.join(CFG_ROOT, tenant, "entity.json") if tenant else None
    gpath = os.path.join(GLOBAL_DIR, "entity.json")
    print(f"DEBUG: load_entity_cfg(tenant={tenant}) -> {tpath or gpath}")
    return _load_entity_cfg_cached(tenant, _mtime(tpath) if tpath else 0.0, _mtime(gpath))

# ---------- SYNONYMS ----------
@lru_cache(maxsize=1)
def load_synonyms_cached():
    return _load_json(SYNONYMS_PATH) or {}

def load_synonyms():
    # hot reload support if 
    NEURONS_CFG_HOT=1
    if CFG_HOT:
        return _load_json(SYNONYMS_PATH) or {}
    return load_synonyms_cached()

# ---------- TIME GRAMMAR ----------
@lru_cache(maxsize=256)
def _load_time_cfg_cached(tenant: str | None, t_mtime: float, g_mtime: float):
    tpath = os.path.join(CFG_ROOT, tenant, "time.json") if tenant else None
    gpath = os.path.join(GLOBAL_DIR, "time.json")
    t = _load_json(tpath) if tpath else None
    g = _load_json(gpath) or {}

    # defaults
    dflt = {
        "relative": {},
        "aliases": {},
        "fiscal": {"enabled": False, "year_start_month": 1}
    }

    # shallow-merge global → tenant (tenant wins if provided)
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

    # final defaults fill
    out["relative"] = out.get("relative", {}) or {}
    out["aliases"]  = out.get("aliases",  {}) or {}
    if "fiscal" not in out: out["fiscal"] = dict(dflt["fiscal"])
    if "enabled" not in out["fiscal"]: out["fiscal"]["enabled"] = False
    if "year_start_month" not in out["fiscal"]: out["fiscal"]["year_start_month"] = 1

    print(f"DEBUG: _load_time_cfg_cached(tenant={tenant}) -> rel={len(out['relative'])}, alias={len(out['aliases'])}, fiscal={out['fiscal']}")
    return out

def load_time_cfg(tenant: str | None):
    tpath = os.path.join(CFG_ROOT, tenant, "time.json") if tenant else None
    gpath = os.path.join(GLOBAL_DIR, "time.json")
    print(f"DEBUG: load_time_cfg(tenant={tenant}) -> {tpath or gpath}")
    return _load_time_cfg_cached(tenant, _mtime(tpath) if tpath else 0.0, _mtime(gpath))
