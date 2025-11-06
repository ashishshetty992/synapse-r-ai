from __future__ import annotations
from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse
import os, json, glob, math, uuid
from typing import Dict, Any, List, Tuple, Optional
from .deps import DATA_DIR, GOLDEN_DIR, ACTIVE_PTR
from .utils_jsonl import read_jsonl
from .models import EvalOut
from .intent import encode_intent_to_vocab
from .synapse import match_candidates, compute_alignplus, score_target_context
from .active_cfg import active_shaping_path
from . import main as service_main

router = APIRouter(prefix="/trainer", tags=["trainer"])

def _rid(request: Optional[Request]) -> str:
    try:
        return getattr(request.state, "request_id")
    except Exception:
        return str(uuid.uuid4())

def _iter_golden(tenant: str | None) -> List[Dict[str, Any]]:
    base = GOLDEN_DIR
    paths = []
    if tenant:
        paths.extend(glob.glob(os.path.join(base, tenant, "*.json")))
    paths.extend(glob.glob(os.path.join(base, "*.json")))
    items = []
    for p in paths:
        try:
            items.append(json.load(open(p)))
        except Exception:
            pass
    return items

def _slot_f1(expected: Dict[str, str], predicted: Dict[str, str]) -> float:
    if not expected and not predicted:
        return 1.0
    if not expected and predicted:
        return 0.0
    if expected and not predicted:
        return 0.0
    exp = set([f"{k}={v}" for k,v in expected.items()])
    got = set([f"{k}={v}" for k,v in predicted.items()])
    tp = len(exp & got)
    fp = len(got - exp)
    fn = len(exp - got)
    prec = tp / (tp + fp) if (tp+fp)>0 else 0.0
    rec  = tp / (tp + fn) if (tp+fn)>0 else 0.0
    if (prec+rec)==0: return 0.0
    return 2*prec*rec/(prec+rec)

def compute_metrics_for_tenant(tenant: str, shaping: Dict[str, Any] | None = None, fewshot: Dict[str, Any] | None = None) -> Dict[str, Any]:
    IEM = service_main.IEM
    if IEM is None:
        return {"top1_acc": 0.0, "mrr": 0.0, "slot_f1": 0.0, "aplus_mean": 0.0, "pathscore_mean": 0.0, "n_golden": 0, "n_feedback": 0}
    gold = _iter_golden(tenant)
    n = len(gold)
    if n == 0:
        n_feedback = len(read_jsonl(os.path.join(DATA_DIR, 'feedback.jsonl')))
        return {"top1_acc": 0.0, "mrr": 0.0, "slot_f1": 0.0, "aplus_mean": 0.0, "pathscore_mean": 0.0, "n_golden": 0, "n_feedback": n_feedback}
    top1_hits = 0
    sum_rr = 0.0
    sum_slot_f1 = 0.0
    sum_aplus = 0.0
    sum_path = 0.0
    for row in gold:
        intent = row.get("intent") or {}
        expected = row.get("expected") or {}
        exp_target = expected.get("target")
        exp_slots  = expected.get("slots") or {}
        intent_vec = encode_intent_to_vocab(intent, IEM.vocab)
        m = match_candidates(IEM, intent_vec, top_k_fields=20, top_k_entities=8, intent_obj=intent)
        ranked = [f"{sf.entity}.{sf.name}" for sf in m.topFields]
        if exp_target in ranked:
            rank = ranked.index(exp_target) + 1
            sum_rr += 1.0 / rank
            if rank == 1: top1_hits += 1
        predicted_slots: Dict[str, str] = {}
        for sf in m.topFields[:8]:
            role = sf.roleTop or "unknown"
            key = role
            predicted_slots.setdefault(key, f"{sf.entity}.{sf.name}")
        sum_slot_f1 += _slot_f1(exp_slots, predicted_slots)
        tgt = exp_target or (ranked[0] if ranked else None)
        if tgt:
            abase = 0.0
            for sf in m.topFields:
                if f"{sf.entity}.{sf.name}" == tgt:
                    abase = float(sf.score); break
            ap = compute_alignplus(abase, intent, tokens=[], mapped_terms=set(), shaping_cfg=shaping or {})
            sum_aplus += ap["Aplus"]
            ctx = score_target_context(IEM, intent_vec, tgt, intent_obj=intent)
            sum_path += float(ctx.get("PathScore", 0.0))
    n_feedback = len(read_jsonl(os.path.join(DATA_DIR, 'feedback.jsonl')))
    return {
        "top1_acc": top1_hits / n,
        "mrr": sum_rr / n,
        "slot_f1": sum_slot_f1 / n,
        "aplus_mean": sum_aplus / n,
        "pathscore_mean": sum_path / n,
        "n_golden": n,
        "n_feedback": n_feedback
    }

@router.get('/last_eval', response_model=EvalOut)
async def last_eval(request: Request = None):
    rid = getattr(request.state, "request_id", "-")
    ckpt = None
    try:
        with open(ACTIVE_PTR, 'r') as f:
            ckpt = json.load(f).get('checkpoint')
    except FileNotFoundError:
        ckpt = None
    metrics = {"top1_acc": 0.0, "mrr": 0.0, "slot_f1": 0.0, "aplus_mean": 0.0, "pathscore_mean": 0.0, "n_golden": 0, "n_feedback": 0}
    if ckpt:
        dirp = os.path.abspath(os.path.join(os.path.dirname(ACTIVE_PTR), "..", "checkpoints", ckpt))
        mpath = os.path.join(dirp, "metrics.json")
        if os.path.exists(mpath):
            try:
                metrics = json.load(open(mpath))
            except Exception:
                pass
    return EvalOut(**metrics, ckpt=ckpt, requestId=rid)

# Compatibility alias router for /eval/last_eval
eval_alias_router = APIRouter(prefix="/eval", tags=["eval"])

@eval_alias_router.get("/last_eval")
def eval_last_eval(request: Request, tenant: Optional[str] = Query(None)):
    """
    Compatibility alias for /trainer/last_eval.
    """
    rid = _rid(request)
    try:
        shaping_path = active_shaping_path()
        
        if not shaping_path or not os.path.exists(shaping_path):
            return JSONResponse(
                {"ok": False, "error": "no active checkpoint or shaping.json missing", "requestId": rid},
                status_code=404
            )
        
        ckpt_dir = os.path.dirname(shaping_path)
        metrics_path = os.path.join(ckpt_dir, "metrics.json")
        
        parts = ckpt_dir.replace("\\", "/").split("/")
        checkpoint = "/".join(parts[-2:]) if len(parts) >= 2 else os.path.basename(ckpt_dir)
        
        if not os.path.exists(metrics_path):
            return {"ok": True, "checkpoint": checkpoint, "metrics": {}, "requestId": rid}
        
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        
        return {"ok": True, "checkpoint": checkpoint, "metrics": metrics, "requestId": rid}
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"last_eval error: {str(e)}", "requestId": rid}, status_code=500)
