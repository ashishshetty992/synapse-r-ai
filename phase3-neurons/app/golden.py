from __future__ import annotations
from fastapi import APIRouter, Request, Body, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import os, json, re, time, uuid
from .deps import GOLDEN_DIR

router = APIRouter(prefix="/golden", tags=["golden"])

def _rid(request: Optional[Request]) -> str:
    try:
        return getattr(request.state, "request_id")
    except Exception:
        return str(uuid.uuid4())

def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9\-]+", "-", (s or "").lower())
    return re.sub(r"-+", "-", s).strip("-") or ("golden-" + uuid.uuid4().hex[:8])

@router.get("/list")
async def golden_list(request: Request = None):
    tenant = getattr(request.state, "tenant_id", None) or "default"
    base = os.path.join(GOLDEN_DIR, tenant)
    items: List[Dict[str, Any]] = []
    if os.path.isdir(base):
        for fn in os.listdir(base):
            if fn.endswith(".json"):
                try:
                    items.append(json.load(open(os.path.join(base, fn))))
                except Exception:
                    pass
    return {"ok": True, "tenant": tenant, "count": len(items), "items": items}

@router.post("/add")
def golden_add(
    request: Request,
    payload: Dict[str, Any] = Body(...)
):
    """
    Body schema:
      {
        "tenant": "acme",                # optional (fallback)
        "item": { "intent": {...}, "expected": {...}, "notes": "..." }
      }
    """
    rid = _rid(request)
    try:
        # Tenant precedence: header → query → body → "default"
        header_tenant = request.headers.get("x-tenant-id")
        query_tenant  = request.query_params.get("tenant")
        body_tenant   = (payload.get("tenant") if isinstance(payload, dict) else None)
        tenant = (header_tenant or query_tenant or body_tenant or "default").strip() or "default"
        
        item = payload.get("item") if isinstance(payload, dict) else None
        if not isinstance(item, dict):
            return JSONResponse({"ok": False, "error": "missing or invalid 'item'", "requestId": rid}, status_code=400)
        
        # Ensure storage under per-tenant folder
        base_dir = os.path.abspath(os.path.join(GOLDEN_DIR, tenant))
        os.makedirs(base_dir, exist_ok=True)
        
        # Filename by ask-type (very simple; adjust as you like)
        ask = item.get("intent", {}).get("ask", "ask")
        fname = f"{ask}-t.json"
        fpath = os.path.join(base_dir, fname)
        
        # Enrich and append item list file
        item["tenant"] = tenant
        item["ts"] = int(time.time() * 1000)
        
        items: list = []
        if os.path.exists(fpath):
            try:
                with open(fpath, "r") as f:
                    items = json.load(f)
                    if not isinstance(items, list):
                        items = []
            except Exception:
                items = []
        
        items.append(item)
        
        with open(fpath, "w") as f:
            json.dump(items, f, indent=2)
        
        return {"ok": True, "tenant": tenant, "file": fpath, "requestId": rid}
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"golden.add error: {str(e)}", "requestId": rid}, status_code=500)

