from __future__ import annotations
from fastapi import APIRouter, Request, Query
from fastapi.responses import JSONResponse
import os, json, glob, math, uuid
from typing import Dict, Any, List, Tuple, Optional

def _score_slots(expected_slots: Dict[str, str], got_slots: Dict[str, str]) -> Tuple[int, int, int, Dict[str, Dict[str, int]]]:
    """
    Generic slot scorer (micro). Returns (tp, fp, fn, per_slot_counts).
    - Exact-string match for each slot key present in expected.
    - We don't penalize for extra slots not in expected (can be toggled later).
    """
    tp = fp = fn = 0
    per_slot: Dict[str, Dict[str, int]] = {}
    exp = expected_slots or {}
    got = got_slots or {}

    for k, v in exp.items():
        per_slot.setdefault(k, {"tp": 0, "fp": 0, "fn": 0})
        got_v = got.get(k)
        if got_v is None:
            fn += 1
            per_slot[k]["fn"] += 1
        elif got_v == v:
            tp += 1
            per_slot[k]["tp"] += 1
        else:
            # slot present but wrong value
            fp += 1
            fn += 1
            per_slot[k]["fp"] += 1
            per_slot[k]["fn"] += 1

    return tp, fp, fn, per_slot
from .deps import DATA_DIR, GOLDEN_DIR, ACTIVE_PTR
from .utils_jsonl import read_jsonl
from .models import EvalOut
from .intent import encode_intent_to_vocab
from .synapse import match_candidates, compute_alignplus, score_target_context
from .active_cfg import active_shaping_path
from . import main as service_main

router = APIRouter(prefix="/eval", tags=["eval"])

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
            obj = json.load(open(p))
            if isinstance(obj, list):
                # file stores a list of items (our current golden_add behavior)
                items.extend([x for x in obj if isinstance(x, dict)])
            elif isinstance(obj, dict):
                items.append(obj)
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
    # slot scoring (micro)
    slot_tp = slot_fp = slot_fn = 0
    per_slot_totals: Dict[str, Dict[str, int]] = {}
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

        # ---- Slot scoring (e.g., timestamp) ----
        # Use synapse_fill to get actual slots from conflict resolution
        from .main import synapse_fill
        from .models import FillRequest
        try:
            fill_req = FillRequest(intent=intent, topKTargets=8)
            fill_resp = synapse_fill(fill_req, request=None)
            got_slots = (fill_resp.get("intentFilled") or {}).get("slots") or {}
        except Exception:
            got_slots = {}
        tp, fp, fn, per_slot = _score_slots(exp_slots, got_slots)
        slot_tp += tp; slot_fp += fp; slot_fn += fn
        # accumulate per-slot breakdown
        for k, c in per_slot.items():
            d = per_slot_totals.setdefault(k, {"tp": 0, "fp": 0, "fn": 0})
            d["tp"] += c["tp"]; d["fp"] += c["fp"]; d["fn"] += c["fn"]
    n_feedback = len(read_jsonl(os.path.join(DATA_DIR, 'feedback.jsonl')))
    
    # -- config snapshot (read-only, safe to log/serialize)
    from . import synapse
    cfg = synapse._CONFIG  # already merged with active shaping if any
    snapshot = {
        "pathScoring": dict(cfg.get("pathScoring", {})),
        # (optional) include more later if you want:
        # "fill": dict(cfg.get("fill", {})),
        # "alignPlus": dict(cfg.get("alignPlus", {})),
        # "weights": dict(cfg.get("weights", {})),
    }
    
    # micro-F1 for slots
    slot_precision = (slot_tp / (slot_tp + slot_fp)) if (slot_tp + slot_fp) > 0 else 0.0
    slot_recall    = (slot_tp / (slot_tp + slot_fn)) if (slot_tp + slot_fn) > 0 else 0.0
    slot_f1 = (2 * slot_precision * slot_recall / (slot_precision + slot_recall)) if (slot_precision + slot_recall) > 0 else 0.0

    top1_acc = top1_hits / max(1, n)
    mrr = sum_rr / max(1, n)
    aplus_mean = sum_aplus / max(1, n)
    pathscore_mean = sum_path / max(1, n)

    return {
        "top1_acc": top1_acc,
        "mrr": mrr,
        "aplus_mean": aplus_mean,
        "pathscore_mean": pathscore_mean,
        "slot_f1": slot_f1,
        "slot_breakdown": per_slot_totals,
        "n_golden": n,
        "n_feedback": n_feedback,
        "snapshot": snapshot,
    }

@router.get('/last_eval', response_model=EvalOut)
async def last_eval(request: Request = None):
    """
    Returns metrics from the active checkpoint's metrics.json file.
    This is different from /trainer/last_eval which reads from canonical last_eval.json files.
    
    This endpoint:
    - Reads the active checkpoint from runtime/active.json
    - Loads metrics.json directly from the checkpoint directory
    - Returns the raw metrics with ckpt field (legacy format for backward compatibility)
    """
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
