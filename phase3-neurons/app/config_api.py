from fastapi import APIRouter, Request, Body
from .active_cfg import active_shaping_path, active_fewshot_path
from . import synapse
from .deps import clamp

router = APIRouter(prefix="/config", tags=["config"])

@router.get("/effective")
def effective_config(request: Request):
    experiment = (request.headers.get("x-experiment") or "").lower().strip()
    source = "tenant" if experiment == "baseline-only" else "active"
    
    path_scoring = dict(synapse._CONFIG.get("pathScoring", {}))
    if isinstance(path_scoring, dict):
        path_scoring["_source"] = source

    align_plus = dict(synapse._CONFIG.get("alignPlus", {}))
    if isinstance(align_plus, dict):
        align_plus["_source"] = source
    
    return {
        "tenant": request.headers.get("x-tenant-id"),
        "experiment": experiment,
        "active": {
            "shaping": active_shaping_path(),
            "fewshot": active_fewshot_path(),
        },
        # THIS is what /synapse/* will actually use right now
        "synapseConfig": {
            "weights": synapse._CONFIG.get("shaping_weights"),
            "fill": synapse._CONFIG.get("fill"),
            "pathScoring": path_scoring,   # includes new schema-aware keys if set
            "entityBlend": synapse._CONFIG.get("entity_blend"),
            "centroidGamma": synapse._CONFIG.get("centroid_gamma"),
            "roleAlphaDefault": synapse._CONFIG.get("role_alpha_default"),
            "alignPlus": align_plus,
        },
    }


# --- Runtime tuning for canarying knobs (process-local, non-persistent) ---
_PS_ALLOWED = {
    # core
    "idPrior": (0.0, 1.0),
    "fkBonus": (0.0, 1.0),
    "lengthPriorBase": (0.5, 0.9999),
    "cosineWeight": (0.0, 1.0),
    # schema-aware
    "pkToFkBonus": (0.0, 1.0),
    "fkToPkBonus": (0.0, 1.0),
    "bridgePenalty": (0.0, 1.0),
    "hubPenalty": (0.0, 1.0),
    "hubDegreeThreshold": (1, 999),
    "fanoutPenalty": (0.0, 1.0),
}

@router.get("/knobs")
def get_knobs():
    """Lightweight view of runtime-tunable knobs."""
    return {
        "ok": True,
        "pathScoring": synapse._CONFIG.get("pathScoring", {}),
        "alignPlus": synapse._CONFIG.get("alignPlus", {}),
    }

@router.post("/knobs")
def set_knobs(body: dict = Body(default={})):
    """
    Update a subset of knobs at runtime (no restart). Safe clamps are applied.
    Body shape (all optional):
      { "pathScoring": {...}, "alignPlus": {"osrZeroWhenNoText": true/false} }
    """
    changed = {}
    ps = body.get("pathScoring") or {}
    
    if isinstance(ps, dict) and ps:
        applied = {}
        for k, v in ps.items():
            if k not in _PS_ALLOWED or v is None:
                continue
            lo, hi = _PS_ALLOWED[k]
            # hubDegreeThreshold is an int; others are floats
            if k == "hubDegreeThreshold":
                vv = int(max(lo, min(hi, int(v))))
            else:
                vv = clamp(f"pathScoring.{k}", float(v))
                # additional hard clamp to bounds
                vv = float(max(lo, min(hi, vv)))
            applied[k] = vv
        if applied:
            current = dict(synapse._CONFIG.get("pathScoring", {}))
            current.update(applied)
            synapse.set_path_scoring(current)
            changed["pathScoring"] = synapse._CONFIG.get("pathScoring", {})
    
    ap = body.get("alignPlus") or {}
    if isinstance(ap, dict) and ("osrZeroWhenNoText" in ap):
        synapse.set_alignplus_cfg({"osrZeroWhenNoText": bool(ap["osrZeroWhenNoText"])})
        changed["alignPlus"] = synapse._CONFIG.get("alignPlus", {})
    
    return {
        "ok": True,
        "changed": changed,
        "now": {
            "pathScoring": synapse._CONFIG.get("pathScoring", {}),
            "alignPlus": synapse._CONFIG.get("alignPlus", {}),
        }
    }

