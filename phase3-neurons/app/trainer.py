from __future__ import annotations
from fastapi import APIRouter, HTTPException, Request, Query
from fastapi.responses import JSONResponse
import os, time, json, glob, copy, uuid
from datetime import datetime
from typing import Optional
from .models import TrainerRunIn, TrainerRunOut, ActivateIn, VersionsOut
from .deps import CKPT_DIR, ACTIVE_PTR, write_json, clamp
from .config import load_shaping_cfg, CFG_ROOT, GLOBAL_DIR
from .eval import compute_metrics_for_tenant
from .active_cfg import active_shaping_path
from . import main as _main

router = APIRouter(prefix="/trainer", tags=["trainer"])

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

    write_json(os.path.join(out_dir, 'shaping.json'), s)
    write_json(os.path.join(out_dir, 'fewshot.json'), _DEFAULT_FEWSHOT)
    try:
        metrics = compute_metrics_for_tenant(tenant=tenant, shaping=s, fewshot=_DEFAULT_FEWSHOT)
    except Exception as _e:
        metrics = {"top1_acc": 0.0, "mrr": 0.0, "slot_f1": 0.0, "aplus_mean": 0.0, "pathscore_mean": 0.0, "n_golden": 0, "n_feedback": 0, "error": str(_e)}
    write_json(os.path.join(out_dir, 'metrics.json'), metrics)
    
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
    Returns metrics for the active checkpoint.
    Looks up runtime/active.json → directory → metrics.json.
    """
    rid = _rid(request)
    try:
        # Resolve active shaping path (points into checkpoints/<tenant>/<ckpt>/shaping.json)
        shaping_path = active_shaping_path()
        
        if not shaping_path or not os.path.exists(shaping_path):
            return JSONResponse(
                {"ok": False, "error": "no active checkpoint or shaping.json missing", "requestId": rid},
                status_code=404
            )
        
        ckpt_dir = os.path.dirname(shaping_path)
        metrics_path = os.path.join(ckpt_dir, "metrics.json")
        
        # Derive checkpoint name from ckpt_dir tail (e.g., 'acme/trainer_YYYYMMDD_HHMM' or 'trainer_...')
        # The tail is usually ".../checkpoints/<tenant>/<ckpt>", so get the last two
        parts = ckpt_dir.replace("\\", "/").split("/")
        checkpoint = "/".join(parts[-2:]) if len(parts) >= 2 else os.path.basename(ckpt_dir)
        
        if not os.path.exists(metrics_path):
            # Not fatal—return empty metrics but 200 so tooling can proceed
            return {
                "ok": True,
                "checkpoint": checkpoint,
                "metrics": {},
                "requestId": rid
            }
        
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        return {
            "ok": True,
            "checkpoint": checkpoint,
            "metrics": metrics,
            "requestId": rid
        }
    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": f"last_eval error: {str(e)}", "requestId": rid},
            status_code=500
        )