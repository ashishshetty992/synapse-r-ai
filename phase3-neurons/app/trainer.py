from __future__ import annotations
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import JSONResponse
from pathlib import Path
import os, time, json, glob, copy, uuid, random
from datetime import datetime
from typing import Optional

random.seed(7)
from .models import TrainerRunIn, TrainerRunOut, ActivateIn, VersionsOut
from .deps import CKPT_DIR, ACTIVE_PTR, write_json, clamp
from .config import load_shaping_cfg, CFG_ROOT, GLOBAL_DIR
from .eval import compute_metrics_for_tenant
from .active_cfg import active_shaping_path
from . import main as _main

router = APIRouter(prefix="/trainer", tags=["trainer"])

CHECKPOINTS_DIR = Path(CKPT_DIR)

def _tenant_dir(tenant: str) -> Path:
    return CHECKPOINTS_DIR / tenant

def _ckpt_dir(tenant: str, ckpt: str) -> Path:
    return CHECKPOINTS_DIR / ckpt if "/" in ckpt else CHECKPOINTS_DIR / tenant / ckpt

def write_last_eval(tenant: str, ckpt: str, metrics: dict):
    """
    Canonical writer:
      - checkpoints/<tenant>/last_eval.json   (tenant-level canonical)
      - checkpoints/<tenant>/<ckpt>/last_eval.json (for reproducibility)
    """
    tdir = _tenant_dir(tenant)
    tdir.mkdir(parents=True, exist_ok=True)
    payload = dict(metrics)
    payload["ckpt"] = ckpt if "/" in ckpt else f"{tenant}/{ckpt}"
    (tdir / "last_eval.json").write_text(json.dumps(payload, indent=2))
    cdir = _ckpt_dir(tenant, ckpt)
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / "last_eval.json").write_text(json.dumps(payload, indent=2))

def read_last_eval(tenant: str) -> dict | None:
    """
    Prefer tenant-level last_eval.json; fall back to newest checkpoint's last_eval.json;
    as a last resort, newest checkpoint's metrics.json.
    """
    tfile = _tenant_dir(tenant) / "last_eval.json"
    if tfile.exists():
        return json.loads(tfile.read_text())
    # newest checkpoint by mtime
    ckpt_glob = str(_tenant_dir(tenant) / "trainer_*")
    cands = sorted(glob.glob(ckpt_glob), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    for c in cands:
        p = Path(c) / "last_eval.json"
        if p.exists():
            return json.loads(p.read_text())
        m = Path(c) / "metrics.json"
        if m.exists():
            payload = json.loads(m.read_text())
            payload["ckpt"] = f"{tenant}/{Path(c).name}"
            return payload
    return None

def _rid(request: Optional[Request]) -> str:
    try:
        return getattr(request.state, "request_id")
    except Exception:
        return str(uuid.uuid4())

_DEFAULT_SHAPING = {
    "weights": {
        "alias_alpha": 0.4,
        "shape_alpha": 0.3,
        "shape_beta":  0.2,
        "metric_alpha":0.4,
        "role_alpha_default":0.3,
    },
    "pathScoring": {
        "idPrior": 0.10,
        "fkBonus": 0.05,
        "lengthPriorBase": 0.92,
        "cosineWeight": 0.75
    },
    "alignPlus": {
        "osrZeroWhenNoText": True
    }
}

_DEFAULT_FEWSHOT = {
    "global": {
        "hints": [
            {"tokens":["gmv","revenue","sales"], "boost":{"role":"money","delta":0.03}},
            {"tokens":["city","municipality","town"], "boost":{"role":"geo","delta":0.02}},
            {"tokens":["status","method","channel","segment"], "boost":{"role":"category","delta":0.02}}
        ],
        "aliases": {"gmv":"revenue"}
    },
    "tenants": {}
}

@router.post('/run', response_model=TrainerRunOut)
async def trainer_run(inp: TrainerRunIn, request: Request = None):
    rid = getattr(request.state, "request_id", "-")
    tenant = (inp.tenant or getattr(request.state, "tenant_id", None) or "_global_").strip()
    
    def _merge(a, b):
        out = copy.deepcopy(a)
        for k, v in (b or {}).items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = _merge(out[k], v)
            else:
                out[k] = v
        return out
    
    s = copy.deepcopy(_DEFAULT_SHAPING)
    try:
        tenant_shaping = None
        if tenant:
            tpath = os.path.join(CFG_ROOT, tenant, "shaping.json")
            if os.path.exists(tpath):
                with open(tpath) as f:
                    tenant_shaping = json.load(f)
        if not tenant_shaping:
            gpath = os.path.join(GLOBAL_DIR, "shaping.json")
            if os.path.exists(gpath):
                with open(gpath) as f:
                    tenant_shaping = json.load(f)
        if tenant_shaping:
            s = _merge(s, tenant_shaping)
    except Exception:
        pass
    
    w = s.get("weights", {})
    for k, v in list(w.items()):
        w[k] = float(clamp(k, v))
    ps = s.get("pathScoring", {})
    for k, v in list(ps.items()):
        ps[k] = float(clamp(f"pathScoring.{k}", v))

    ckpt_name = f"trainer_{datetime.now().strftime('%Y%m%d_%H%M')}"
    out_dir = os.path.join(CKPT_DIR, tenant, ckpt_name)
    os.makedirs(out_dir, exist_ok=True)

    # Latin-hypercube search over parameter grid
    candidates = []
    grid = [
        ("weights.alias_alpha", [0.30, 0.35, 0.40, 0.45]),
        ("weights.metric_alpha", [0.10, 0.12, 0.15]),
        ("pathScoring.cosineWeight", [0.70, 0.75, 0.80]),
        ("pathScoring.lengthPriorBase", [0.90, 0.92, 0.94]),
        ("salience.gamma", [0.10, 0.15, 0.20]),
    ]
    samples = min(int(inp.maxGrid or 16), 32)
    for _ in range(samples):
        cand = copy.deepcopy(s)
        for k, vals in grid:
            tgt = cand
            parts = k.split(".")
            for p in parts[:-1]:
                tgt = tgt.setdefault(p, {})
            tgt[parts[-1]] = random.choice(vals)
        candidates.append(cand)

    best = {"metrics": {"top1_acc": 0, "mrr": 0}, "cfg": s}
    for cfg in candidates:
        try:
            m = compute_metrics_for_tenant(tenant=tenant, shaping=cfg, fewshot=_DEFAULT_FEWSHOT)
            if (m.get("top1_acc", 0), m.get("mrr", 0)) > (best["metrics"].get("top1_acc", 0), best["metrics"].get("mrr", 0)):
                best = {"metrics": m, "cfg": cfg}
        except Exception as _e:
            continue

    s = best["cfg"]
    metrics = best["metrics"]
    principles = {"enabled": {"unknownBackoff": True, "hubPenalty": True}, "notes": "auto-toggle v1"}

    write_json(os.path.join(out_dir, 'shaping.json'), s)
    write_json(os.path.join(out_dir, 'fewshot.json'), _DEFAULT_FEWSHOT)
    write_json(os.path.join(out_dir, 'principles.json'), principles)
    write_json(os.path.join(out_dir, 'metrics.json'), metrics)
    
    # Write canonical last_eval.json (tenant-level + checkpoint-scoped)
    write_last_eval(tenant, ckpt_name, metrics)
    
    try:
        if getattr(_main, "PROM_REG", None):
            tenant_lbl = tenant
            _main.MET_TRAIN_RUNS.inc()
            _main.MET_TOP1.labels(tenant=tenant_lbl).set(float(metrics.get("top1_acc", 0.0)))
            _main.MET_MRR.labels(tenant=tenant_lbl).set(float(metrics.get("mrr", 0.0)))
            _main.MET_GOLDEN.labels(tenant=tenant_lbl).set(float(metrics.get("n_golden", 0)))
    except Exception:
        pass
    
    return TrainerRunOut(ok=True, ckpt=os.path.join(tenant, ckpt_name), metrics=metrics, requestId=rid)

@router.post('/activate')
async def trainer_activate(inp: ActivateIn, request: Request = None):
    rid = getattr(request.state, "request_id", "-")
    dirp = os.path.join(CKPT_DIR, inp.checkpoint)
    shp = os.path.join(dirp, 'shaping.json')
    fs = os.path.join(dirp, 'fewshot.json')
    
    if not os.path.isdir(dirp) or not os.path.exists(shp) or not os.path.exists(fs):
        return {"ok": False, "error": "checkpoint incomplete or not found", "checkpoint": inp.checkpoint, "requestId": rid}
    
    active = {
        "checkpoint": inp.checkpoint,
        "shaping": shp,
        "fewshot": fs
    }
    os.makedirs(os.path.dirname(ACTIVE_PTR), exist_ok=True)
    write_json(ACTIVE_PTR, active)
    return {"ok": True, "active": active, "requestId": rid}

@router.get('/versions', response_model=VersionsOut)
async def trainer_versions(request: Request = None):
    rid = getattr(request.state, "request_id", "-")
    items = []
    if not os.path.isdir(CKPT_DIR):
        return {"versions": items, "requestId": rid}
    
    for name in sorted(os.listdir(CKPT_DIR)):
        d = os.path.join(CKPT_DIR, name)
        if not os.path.isdir(d):
            continue
        rec = {"checkpoint": name, "files": []}
        for f in ("shaping.json", "fewshot.json", "metrics.json"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                rec["files"].append(f)
        m = os.path.join(d, "metrics.json")
        if os.path.exists(m):
            try:
                rec["metrics"] = json.load(open(m))
            except:
                pass
        items.append(rec)
    return {"versions": items, "requestId": rid}

@router.get("/last_eval")
def trainer_last_eval(request: Request, tenant: Optional[str] = Query(None)):
    """
    Returns metrics from canonical tenant-level last_eval.json.
    Falls back to newest checkpoint's last_eval.json or metrics.json.
    """
    rid = _rid(request)
    try:
        # Get tenant from query param, request state, or default
        tenant_id = tenant or getattr(request.state, "tenant_id", None) or "acme"
        payload = read_last_eval(tenant_id)
        if payload is None:
            return {
                "ok": None,
                "checkpoint": None,
                "top1_acc": None,
                "mrr": None,
                "slot_f1": None,
                "slot_breakdown": {},
                "aplus_mean": None,
                "pathscore_mean": None,
                "snapshot": {},
                "requestId": rid
            }
        # Ensure consistent response format with ok and checkpoint fields
        # Create response with ok and checkpoint, excluding ckpt from payload to avoid duplication
        response = dict(payload)
        ckpt_value = response.pop("ckpt", None)
        # Remove requestId if it exists in payload (we'll add it at the end)
        response.pop("requestId", None)
        return {
            "ok": True,
            "checkpoint": ckpt_value,
            **response,  # Spread all other fields (top1_acc, mrr, etc.)
            "requestId": rid
        }
    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": f"last_eval error: {str(e)}", "requestId": rid},
            status_code=500
        )