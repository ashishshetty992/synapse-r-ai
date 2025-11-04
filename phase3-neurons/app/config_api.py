# phase3-neurons/app/config_api.py
from fastapi import APIRouter, Request
import json
from .active_cfg import active_shaping_path, active_fewshot_path, _read_active

router = APIRouter(prefix="/config", tags=["config"])

@router.get("/effective")
def effective_config(request: Request = None):
    rid = getattr(request.state, "request_id", "-")
    a = _read_active() or {}
    shp = active_shaping_path()
    fs = active_fewshot_path()
    
    return {
        "active": {
            "checkpoint": a.get("checkpoint"),
            "shaping": shp,
            "fewshot": fs,
            "exists": {
                "shaping": bool(shp),
                "fewshot": bool(fs)
            }
        },
        "requestId": rid
    }

