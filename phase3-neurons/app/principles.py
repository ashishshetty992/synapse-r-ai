from __future__ import annotations
from fastapi import APIRouter, Request
import json, os
from .deps import ACTIVE_PTR

router = APIRouter(prefix="/principles", tags=["principles"])

@router.get("/show")
def show(request: Request = None):
    rid = getattr(request.state, "request_id", "-")
    try:
        ptr = json.load(open(ACTIVE_PTR))
        p = os.path.join(os.path.dirname(ptr["shaping"]), "principles.json")
        merged = json.load(open(p)) if os.path.exists(p) else {"enabled":{}}
        return {"ok": True, "merged": merged, "requestId": rid}
    except Exception as e:
        return {"ok": False, "error": str(e), "requestId": rid}