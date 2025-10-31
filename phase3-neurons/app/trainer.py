# phase3-neurons/app/trainer.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
import os, time, json, glob
from datetime import datetime
from .models import TrainerRunIn, TrainerRunOut, ActivateIn, VersionsOut
from .deps import CKPT_DIR, ACTIVE_PTR, write_json, clamp

router = APIRouter(prefix="/trainer", tags=["trainer"])

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
            {"tokens":["gmv","revenue"], "boost":{"role":"money","delta":0.03}}
        ],
        "aliases": {"gmv":"revenue"}
    },
    "tenants": {}
}

@router.post('/run', response_model=TrainerRunOut)
async def trainer_run(inp: TrainerRunIn):
    # (stub) pretend we searched; return default but clamp
    s = _DEFAULT_SHAPING.copy()
    w = s["weights"]
    for k,v in list(w.items()):
        w[k] = float(clamp(k, v))
    ps = s["pathScoring"]
    for k,v in list(ps.items()):
        ps[k] = float(clamp(f"pathScoring.{k}", v))

    ckpt_name = f"trainer_{datetime.now().strftime('%Y%m%d_%H%M')}"
    out_dir = os.path.join(CKPT_DIR, ckpt_name)
    os.makedirs(out_dir, exist_ok=True)

    write_json(os.path.join(out_dir, 'shaping.json'), s)
    write_json(os.path.join(out_dir, 'fewshot.json'), _DEFAULT_FEWSHOT)
    metrics = {  # placeholder, to be replaced by real evaluator numbers
        "top1_acc": 0.0, "mrr": 0.0, "slot_f1": 0.0,
        "aplus_mean": 0.0, "pathscore_mean": 0.0
    }
    write_json(os.path.join(out_dir, 'metrics.json'), metrics)
    return TrainerRunOut(ok=True, ckpt=ckpt_name, metrics=metrics)

@router.post('/activate')
async def trainer_activate(inp: ActivateIn):
    dirp = os.path.join(CKPT_DIR, inp.checkpoint)
    shp = os.path.join(dirp, 'shaping.json')
    fs = os.path.join(dirp, 'fewshot.json')
    
    if not os.path.isdir(dirp) or not os.path.exists(shp) or not os.path.exists(fs):
        return {"ok": False, "error": "checkpoint incomplete or not found", "checkpoint": inp.checkpoint}
    
    active = {
        "checkpoint": inp.checkpoint,
        "shaping": shp,
        "fewshot": fs
    }
    os.makedirs(os.path.dirname(ACTIVE_PTR), exist_ok=True)
    write_json(ACTIVE_PTR, active)
    return {"ok": True, "active": active}

@router.get('/versions', response_model=VersionsOut)
async def trainer_versions():
    items = []
    if not os.path.isdir(CKPT_DIR):
        return {"versions": items}
    
    for name in sorted(os.listdir(CKPT_DIR)):
        d = os.path.join(CKPT_DIR, name)
        if not os.path.isdir(d):
            continue
        rec = {"checkpoint": name, "files": []}
        for f in ("shaping.json", "fewshot.json", "metrics.json"):
            p = os.path.join(d, f)
            if os.path.exists(p):
                rec["files"].append(f)
        # optional metrics
        m = os.path.join(d, "metrics.json")
        if os.path.exists(m):
            try:
                rec["metrics"] = json.load(open(m))
            except:
                pass
        items.append(rec)
    return {"versions": items}

@router.get('/last_eval')
async def trainer_last_eval():
    from .deps import read_json
    # read ACTIVE_PTR's metrics if present; otherwise empty
    try:
        active = read_json(ACTIVE_PTR)
    except:
        return {"top1_acc": 0.0, "mrr": 0.0, "slot_f1": 0.0, "aplus_mean": 0.0, "pathscore_mean": 0.0, "n_golden": 0, "n_feedback": 0, "ckpt": None}
    
    m = os.path.join(os.path.dirname(active["shaping"]), "metrics.json")
    if os.path.exists(m):
        metrics = read_json(m)
        return {
            "top1_acc": metrics.get("top1_acc", 0.0),
            "mrr": metrics.get("mrr", 0.0),
            "slot_f1": metrics.get("slot_f1", 0.0),
            "aplus_mean": metrics.get("aplus_mean", 0.0),
            "pathscore_mean": metrics.get("pathscore_mean", 0.0),
            "n_golden": metrics.get("n_golden", 0),
            "n_feedback": metrics.get("n_feedback", 0),
            "ckpt": active.get("checkpoint")
        }
    
    return {"top1_acc": 0.0, "mrr": 0.0, "slot_f1": 0.0, "aplus_mean": 0.0, "pathscore_mean": 0.0, "n_golden": 0, "n_feedback": 0, "ckpt": active.get("checkpoint")}