from fastapi import FastAPI, Body, Request, Query
from fastapi.responses import JSONResponse, Response
from typing import Optional, Dict, Any, List
import os, json, uuid, logging, time
from collections import deque, defaultdict
from math import sqrt

try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logging.info(f"Loaded .env from {env_path}")
    else:
        load_dotenv()
except ImportError:
    pass
except Exception as e:
    logging.warning(f"Failed to load .env: {e}")

from .feedback import router as feedback_router
from .eval import router as eval_router, eval_alias_router
from .trainer import router as trainer_router
from .fewshot import router as fewshot_router
from .config_api import router as config_router
from .golden import router as golden_router

from .models import (
    BuildIEMRequest, BuildIEMResponse,
    IntentRequest, IntentEncoding,
    MatchRequest, MatchResponse, ScoredField, ScoredEntity,
    VerifyIEMResponse, RoleBoostStat, DriftAlert, ClusterQuality as CQ,
    NLRequest, IntentEncodingNL,
    FillRequest, FillResponse, ConflictNote, AlignPlus,
    PathsRequest, PathsResponse, PathCandidate, PathEdge,
    ExplainRequest, ExplainResponse, ExplainItem, CandidateScore,
    GenerateRequest, GenerateExampleResponse, GenIQL, CoverageStat,
    UEM
)
from .iem import IEMIndex, build_iem_from_uem, compute_role_boost, compute_alias_stability, cluster_quality
from .intent import encode_intent_to_vocab, encode_intent_nl
from .synapse import match_candidates, compute_alignplus, _parse_entity_field, _build_join_graph, _bfs_paths, _path_score, score_target_context, resolve_conflicts
from .ltr import ltr_rerank_paths, build_degree_index, _path_feats
from .config import load_role_cfg, load_shaping_cfg, load_entity_cfg, load_synonyms, load_time_cfg, CFG_HOT
from .generator import generate_all, write_examples, write_coverage
from . import synapse
from .synapse import set_role_config, set_shaping_weights
from .deps import ACTIVE_PTR, clamp
from .decision_log import log_decision

try:
    from prometheus_client import CollectorRegistry, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROM_REG = CollectorRegistry()
    MET_TRAIN_RUNS = Gauge("synapse_trainer_runs_total", "Trainer runs", registry=PROM_REG)
    MET_TOP1 = Gauge("synapse_trainer_top1_acc", "Top1 accuracy (last run)", ["tenant"], registry=PROM_REG)
    MET_MRR = Gauge("synapse_trainer_mrr", "MRR (last run)", ["tenant"], registry=PROM_REG)
    MET_FEEDBACK = Gauge("synapse_feedback_rows_total", "Feedback rows", registry=PROM_REG)
    MET_GOLDEN = Gauge("synapse_golden_rows_total", "Golden rows", ["tenant"], registry=PROM_REG)
except Exception:
    PROM_REG = None
    generate_latest = None
    CONTENT_TYPE_LATEST = None

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(request_id)s] %(message)s",
    level=logging.INFO
)

class RequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


def log_json(level: str, rid: str, event: str, **kwargs):
    payload = {"ts": time.time(), "level": level, "rid": rid, "event": event, **kwargs}
    logger.log(getattr(logging, level.upper(), logging.INFO), json.dumps(payload), extra={"request_id": rid})

def ok(rid: str, payload: dict) -> dict:
    return {"ok": True, "requestId": rid, **payload}

def err(status: int, rid: str, msg: str) -> JSONResponse:
    return JSONResponse({"ok": False, "error": msg, "requestId": rid}, status_code=status)

# ==== NEW: per-edge scoring helpers (non-invasive, no external deps) ====

def _lookup_field(iem, entity: str, name: str):
    """Return the IEM field object (with .vec and .role) or None."""
    try:
        # IEMIndex uses a list of Field objects: (entity, name, vec, role, ...)
        for f in iem.fields:
            if f.entity == entity and f.name == name:
                return f
    except Exception:
        pass
    return None

def _cosine(u, v) -> float:
    """Cosine similarity for two equal-length lists; safe if None/mismatch."""
    if not u or not v or len(u) != len(v):
        return 0.0
    dot = 0.0
    nu = 0.0
    nv = 0.0
    for a, b in zip(u, v):
        dot += a * b
        nu += a * a
        nv += b * b
    if nu <= 0.0 or nv <= 0.0:
        return 0.0
    return float(dot / (sqrt(nu) * sqrt(nv)))

def _edge_breakdown(iem, intent_vec, edge, hop_idx: int, ps_cfg: dict, deg_index: dict[str,int]) -> dict:
    """
    Mirror synapse._path_score edge math for debug:
      - baseCos: 0.5*(cos(intent, src) + cos(intent, dst)), clamped >= 0
      - basePrior: idPrior_if_*_id + fkBonus_if_fk_or_fk_rev + dir bonuses + hub/fanout penalties
      - blend: cosWeight*baseCos + (1-cosWeight)*basePrior   (length prior is applied at PATH level)
    """
    se, sf, de, df, why = edge
    # look up fields
    src_field = _lookup_field(iem, se, sf)
    dst_field = _lookup_field(iem, de, df)
    
    # cosine(intent, src/dst)
    def _safe_cos(f):
        return max(0.0, _cosine(intent_vec, getattr(f, "vec", None))) if f else 0.0
    
    ca = _safe_cos(src_field)
    cb = _safe_cos(dst_field)
    base_cos = (ca + cb) * 0.5
    
    # knobs
    id_prior     = float(ps_cfg.get("idPrior", 0.10))
    fk_bonus     = float(ps_cfg.get("fkBonus", 0.05))
    cos_w        = float(ps_cfg.get("cosineWeight", 0.75))
    pk_fk_bonus  = float(ps_cfg.get("pkToFkBonus", 0.00))
    fk_pk_bonus  = float(ps_cfg.get("fkToPkBonus", 0.00))
    hub_pen      = float(ps_cfg.get("hubPenalty", 0.00))
    hub_thresh   = int(ps_cfg.get("hubDegreeThreshold", 8))
    fanout_pen   = float(ps_cfg.get("fanoutPenalty", 0.00))
    
    # prior (string-based like _path_score)
    base_prior = 0.0
    if sf.endswith("_id") or df.endswith("_id"):
        base_prior += id_prior
    if why in {"fk", "fk_rev"}:
        base_prior += fk_bonus
        # direction-aware nudges (same rule as _path_score)
        if why == "fk":
            base_prior += pk_fk_bonus
        else:  # fk_rev
            base_prior += fk_pk_bonus
    
    # penalties depend on dst entity degree (same as _path_score)
    dst_deg = max(0, deg_index.get(de, 0))
    if dst_deg >= hub_thresh:
        base_prior -= hub_pen
    if fanout_pen:
        base_prior -= fanout_pen * max(0, dst_deg - 1)
    
    blend = (cos_w * base_cos) + ((1.0 - cos_w) * base_prior)
    
    return {
        "edge": f"{se}.{sf} \u2192 {de}.{df}",
        "why": why,
        "hop": hop_idx + 1,
        "baseCos": round(base_cos, 6),
        "basePrior": round(base_prior, 6),
        "blend": round(blend, 6),
        "notes": {
            "idPriorIf_*_id": id_prior,
            "fkBonusIfFkOrRev": fk_bonus,
            "dirBonus(pkToFk,fkToPk)": [pk_fk_bonus, fk_pk_bonus],
            "hubPen": hub_pen,
            "hubThresh": hub_thresh,
            "fanoutPenPerExtra": fanout_pen,
        }
    }

def _path_edge_breakdowns(iem, intent_vec, raw_edges, ps_cfg: dict, deg_index: dict[str,int]) -> list[dict]:
    """Return a list of per-edge debug dicts for the given raw path."""
    out = []
    for idx, e in enumerate(raw_edges):
        try:
            out.append(_edge_breakdown(iem, intent_vec, e, idx, ps_cfg, deg_index))
        except Exception as _e:
            se, sf, de, df, why = e
            out.append({
                "edge": f"{se}.{sf} \u2192 {de}.{df}",
                "error": str(_e),
                "why": why,
                "hop": idx + 1
            })
    return out

# ==== END NEW helpers ====

_rate_buckets: dict[str, deque] = defaultdict(deque)

app = FastAPI(title="SYNAPSE-R Phase 3 Neurons", version="0.1.0")

app.include_router(feedback_router)
app.include_router(eval_router)
app.include_router(eval_alias_router)
app.include_router(trainer_router)
app.include_router(fewshot_router)
app.include_router(config_router)
app.include_router(golden_router)

logger = logging.getLogger("neurons")
logger.addFilter(RequestIdFilter())

@app.middleware("http")
async def guardrails_mw(request: Request, call_next):
    rid = str(uuid.uuid4())
    request.state.request_id = rid
    
    # Reset to baseline at the start of every request (prevents cross-request bleed)
    _apply_baseline_cfg()

    if TRUST_PROXY and request.headers.get("x-forwarded-for"):
        ip = request.headers["x-forwarded-for"].split(",")[0].strip()
    else:
        ip = request.client.host if request.client else "unknown"
    
    ua = request.headers.get("user-agent")

    tenant = request.headers.get("x-tenant-id") or request.query_params.get("tenant")
    request.state.tenant_id = tenant
    request.state.experiment = (request.headers.get("x-experiment") or "").lower().strip()

    try:
        # Hot reload: clear caches if enabled
        if CFG_HOT:
            from .config import _load_role_cfg_cached, _load_entity_cfg_cached, _load_time_cfg_cached, load_synonyms_cached
            _load_role_cfg_cached.cache_clear()
            _load_entity_cfg_cached.cache_clear()
            _load_time_cfg_cached.cache_clear()
            load_synonyms_cached.cache_clear()
        
        # Compute experiment mode first
        apply_active = (request.state.experiment != "baseline-only")
        
        # 1) Load tenant/global cfg (respect allow_active flag)
        rc = load_role_cfg(tenant)
        sc = load_shaping_cfg(tenant, allow_active=apply_active)
        ec = load_entity_cfg(tenant)

        log_json("info", rid, "cfg.load.start", tenant=tenant, 
                role_cfg_keys=list(rc.keys()) if rc else [],
                shaping_cfg_keys=list(sc.keys()) if sc else [],
                entity_cfg_keys=list(ec.keys()) if ec else [])

        if isinstance(rc.get("keywords"), dict):
            synapse.set_role_config(rc["keywords"])
            log_json("info", rid, "cfg.role.applied", keyword_count=len(rc["keywords"]))

        allow_apply = (not TRAINER_TENANT_CANARY) or (tenant and tenant in TRAINER_TENANT_CANARY)
        
        log_json("debug", rid, "cfg.apply.flags", 
                apply_active=apply_active,
                allow_apply=allow_apply,
                sc_pathScoring=sc.get("pathScoring") if isinstance(sc, dict) else None)
        
        # Apply shaping config from sc (which is either tenant/global or active checkpoint based on allow_active)
        # Note: load_shaping_cfg already handles the active checkpoint when allow_active=True
        if allow_apply and isinstance(sc.get("weights"), dict):
            synapse.set_shaping_weights(sc["weights"])
            if sc["weights"].get("role_alpha_default") is not None:
                synapse.set_role_alpha_default(float(sc["weights"]["role_alpha_default"]))
        
        if isinstance(sc.get("fill"), dict):
            synapse.set_fill_config(sc["fill"])
        
        if allow_apply and isinstance(sc.get("pathScoring"), dict):
            synapse.set_path_scoring(sc["pathScoring"])
        
        if isinstance(sc.get("logging"), dict):
            synapse.set_logging_cfg(sc["logging"])
        
        if isinstance(sc.get("alignPlus"), dict):
            synapse.set_alignplus_cfg(sc["alignPlus"])
        
        # OPTIONAL: backward-compat shim (only if you still want to accept the old fill flag)
        if "fill" in sc and isinstance(sc["fill"], dict):
            compat = sc["fill"].get("osrZeroWhenNoText")
            if compat is not None and "alignPlus" not in sc:
                # don't override if alignPlus already provided
                synapse.set_alignplus_cfg({"osrZeroWhenNoText": bool(compat)})
        
        # Apply entity config (always, regardless of canary)
        eb = (ec.get("entity_blend") or {})
        synapse.set_entity_blend(
            entity_weight=float(eb.get("entityWeight", 0.70)),
            field_weight=float(eb.get("fieldWeight", 0.30)),
            entity_alias_alpha=float(eb.get("entityAliasAlpha", 0.15)),
        )
        synapse.set_centroid_gamma(float(ec.get("centroid_gamma", 0.60)))
        if not (allow_apply and isinstance(sc.get("weights"), dict) and sc["weights"].get("role_alpha_default") is not None):
            synapse.set_role_alpha_default(float(ec.get("role_alpha", 0.12)))
        
        log_json("info", rid, "cfg.entity.applied", 
                entity_weight=eb.get("entityWeight", 0.70),
                field_weight=eb.get("fieldWeight", 0.30),
                entity_alias_alpha=eb.get("entityAliasAlpha", 0.15),
                centroid_gamma=ec.get("centroid_gamma", 0.60),
                role_alpha=ec.get("role_alpha", 0.12))
        
        if not apply_active:
            log_json("info", rid, "cfg.baseline.only", tenant=tenant, 
                    pathScoring_from=("tenant/global" if tenant else "global"))
        
        # Sanity log: what was actually applied
        log_json("info", rid, "cfg.final.synapse",
                pathScoring=synapse._CONFIG.get("pathScoring"),
                weights=synapse._CONFIG.get("shaping_weights"),
                fill_conflict=synapse._CONFIG.get("fill", {}).get("conflict"))
                
    except Exception as e:
        log_json("warning", rid, "cfg.load.error", error=str(e))

    if API_KEY:
        given = request.headers.get("x-api-key")
        if given != API_KEY:
            log_json("warning", rid, "auth.denied", path=str(request.url), method=request.method, ip=ip)
            return JSONResponse({"ok": False, "error": "unauthorized", "requestId": rid}, status_code=401)

    now = time.time()
    bucket = _rate_buckets[ip]
    while bucket and now - bucket[0] > RL_WINDOW_SEC:
        bucket.popleft()
    if len(bucket) >= RL_MAX:
        reset = RL_WINDOW_SEC - int(now - bucket[0])
        headers = {
            "x-request-id": rid,
            "x-ratelimit-limit": str(RL_MAX),
            "x-ratelimit-remaining": "0",
            "x-ratelimit-reset": str(reset),
        }
        log_json("warning", rid, "ratelimit.block", path=str(request.url), method=request.method, ip=ip, limit=RL_MAX, window=RL_WINDOW_SEC)
        return JSONResponse({"ok": False, "error": "rate_limited", "requestId": rid}, status_code=429, headers=headers)
    bucket.append(now)

    try:
        clen = request.headers.get("content-length")
        if clen and int(clen) > int(os.environ.get("NEURONS_MAX_BODY", "2097152")):
            return err(413, rid, "payload_too_large")
    except Exception:
        pass

    t0 = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        log_json("error", rid, "unhandled.error", path=str(request.url), method=request.method, ip=ip, error=str(e))
        raise
    duration_ms = int((time.time() - t0) * 1000)
    remaining = max(0, RL_MAX - len(bucket))
    response.headers["x-request-id"] = rid
    response.headers["x-ratelimit-limit"] = str(RL_MAX)
    response.headers["x-ratelimit-remaining"] = str(remaining)
    log_json("info", rid, "http.access", path=str(request.url), method=request.method, status=getattr(response, "status_code", 200), ip=ip, ua=ua, dur_ms=duration_ms, tenant=getattr(request.state, "tenant_id", None))
    return response

UEM_PATH = os.environ.get("UEM_PATH", os.path.join(os.getcwd(), "uem.json"))
IEM_PATH = os.environ.get("IEM_PATH", os.path.join(os.getcwd(), "iem.json"))

ROLE_CFG_PATH = os.environ.get("NEURONS_ROLE_CFG")
SHAPING_CFG_PATH = os.environ.get("NEURONS_SHAPING_CFG")

API_KEY = os.environ.get("NEURONS_API_KEY")
RL_MAX = int(os.environ.get("NEURONS_RL_MAX", "60"))
RL_WINDOW_SEC = int(os.environ.get("NEURONS_RL_WINDOW", "60"))
TRUST_PROXY = os.environ.get("NEURONS_TRUST_PROXY", "0") == "1"
OUT_BASE = os.path.abspath(os.environ.get("NEURONS_OUT_BASE", os.getcwd()))
TRAINER_SHADOW_ONLY = os.environ.get("TRAINER_SHADOW_ONLY", "false").lower() == "true"
TRAINER_TENANT_CANARY = {t.strip() for t in os.environ.get("TRAINER_TENANT_CANARY", "").split(",") if t.strip()}
LTR_DEFAULT = os.environ.get("NEURONS_LTR_DEFAULT", "0").lower() in {"1","true","yes"}
print(f"DEBUG: TRAINER_TENANT_CANARY={TRAINER_TENANT_CANARY}")

IEM: Optional[IEMIndex] = None
COVERAGE_CACHE: Optional[List[Dict[str, Any]]] = None

_BASELINE = {
    "weights": {"alias_alpha": 0.35, "shape_alpha": 0.12, "shape_beta": 0.10, "metric_alpha": 0.12},
    "entity_blend": {"entityWeight": 0.70, "fieldWeight": 0.30, "entityAliasAlpha": 0.15},
    "centroid_gamma": 0.60,
    "role_alpha_default": 0.12,
    "pathScoring": {"idPrior": 0.10, "fkBonus": 0.05, "lengthPriorBase": 0.92, "cosineWeight": 0.75},
}

def _apply_baseline_cfg():
    synapse.set_shaping_weights(_BASELINE["weights"])
    synapse.set_entity_blend(**_BASELINE["entity_blend"])
    synapse.set_centroid_gamma(_BASELINE["centroid_gamma"])
    synapse.set_role_alpha_default(_BASELINE["role_alpha_default"])
    synapse.set_path_scoring(_BASELINE["pathScoring"])


def _load_uem_from_path(path: str) -> UEM:
    with open(path, "r") as f:
        data = json.load(f)
    return UEM.model_validate(data)


def _save_json(path: str, data: Dict[str, Any]):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _load_json_if(path: Optional[str]) -> Optional[dict]:
    if not path: return None
    if not os.path.exists(path): return None
    with open(path, "r") as f:
        return json.load(f)


@app.on_event("startup")
def boot():
    global IEM
    try:
        if os.path.exists(IEM_PATH):
            IEM = IEMIndex.load(IEM_PATH)
        elif os.path.exists(UEM_PATH):
            uem = _load_uem_from_path(UEM_PATH)
            iem = build_iem_from_uem(uem, dim=256)
            IEMIndex.save_json(IEM_PATH, iem.to_json())
            IEM = iem
    except Exception as e:
        logger.warning(f"Startup warning: {e}")
    
    role_cfg = _load_json_if(ROLE_CFG_PATH)
    if role_cfg and isinstance(role_cfg.get("keywords"), dict):
        set_role_config(role_cfg["keywords"])
    shaping_cfg = _load_json_if(SHAPING_CFG_PATH)
    if shaping_cfg and isinstance(shaping_cfg.get("weights"), dict):
        set_shaping_weights(shaping_cfg["weights"])


@app.get("/metrics")
def prometheus_metrics():
    """Prometheus metrics endpoint."""
    if PROM_REG is None or generate_latest is None or CONTENT_TYPE_LATEST is None:
        return Response(content="# Prometheus client not available\n", media_type="text/plain")
    return Response(content=generate_latest(PROM_REG), media_type=CONTENT_TYPE_LATEST)

@app.get("/healthz")
def healthz():
    return {"ok": True, "has_iem": IEM is not None}

@app.get("/debug/runtime")
def debug_runtime(request: Request = None):
    """Debug endpoint showing current runtime configuration."""
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    try:
        active_ptr_data = None
        if os.path.exists(ACTIVE_PTR):
            try:
                with open(ACTIVE_PTR, "r") as f:
                    active_ptr_data = json.load(f)
            except Exception as e:
                active_ptr_data = {"error": str(e)}
        
        from .active_cfg import active_shaping_path, active_fewshot_path
        
        return {
            "ok": True,
            "requestId": rid,
            "environment": {
                "trainerShadowOnly": TRAINER_SHADOW_ONLY,
                "trainerTenantCanary": sorted(list(TRAINER_TENANT_CANARY)),
                "cfgHot": CFG_HOT,
            },
            "activeCheckpoint": {
                "ptrPath": ACTIVE_PTR,
                "ptrExists": os.path.exists(ACTIVE_PTR),
                "ptrData": active_ptr_data,
                "shapingPath": active_shaping_path(),
                "fewshotPath": active_fewshot_path(),
            },
            "synapseRuntimeConfig": {
                "weights": synapse._CONFIG.get("shaping_weights"),
                "fill": synapse._CONFIG.get("fill"),
                "pathScoring": synapse._CONFIG.get("pathScoring"),
                "entityBlend": synapse._CONFIG.get("entity_blend"),
                "centroidGamma": synapse._CONFIG.get("centroid_gamma"),
                "roleAlphaDefault": synapse._CONFIG.get("role_alpha_default"),
            },
        }
    except Exception as e:
        return err(500, rid, f"Debug runtime error: {str(e)}")

@app.get("/debug/graph")
def debug_graph(request: Request = None,
                top: int = Query(10, ge=1, le=100),
                thresh: int = Query(6, ge=1)):
    """Lightweight schema graph stats: degrees, top hubs, fanout summary."""
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    if IEM is None:
        return err(400, rid, "IEM not loaded. Call /iem/build first.")
    try:
        deg = build_degree_index(IEM)
        hubs = sorted(deg.items(), key=lambda kv: kv[1], reverse=True)[:top]
        over = [(e, d) for e, d in deg.items() if d >= thresh]
        return {
            "ok": True,
            "requestId": rid,
            "entityCount": len(getattr(IEM, "entities", []) or deg.keys()),
            "joinCount": len(getattr(IEM, "joins", []) or []),
            "degree": deg,
            "topHubs": hubs,
            "fanoutSummary": {
                "threshold": thresh,
                "countAtOrAbove": len(over),
                "examples": sorted(over, key=lambda kv: kv[1], reverse=True)[:min(top, len(over))]
            }
        }
    except Exception as e:
        return err(500, rid, f"Debug graph error: {str(e)}")


@app.post("/iem/build", response_model=BuildIEMResponse)
def iem_build(req: BuildIEMRequest = Body(...), request: Request = None):
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    try:
        global IEM
        if req.dim is not None and (req.dim < 32 or req.dim > 2048):
            return err(400, rid, "dim out of range [32,2048]")

        if req.uem is not None:
            uem = req.uem
        else:
            path = req.uemPath or UEM_PATH
            if not os.path.exists(path):
                return err(400, rid, f"UEM file not found: {path}")
            uem = _load_uem_from_path(path)

        dim = req.dim or 256
        iem = build_iem_from_uem(uem, dim=dim)

        out_path = req.iemPath or IEM_PATH
        try:
            if os.path.exists(out_path):
                os.replace(out_path, out_path + ".bak")
                log_json("info", rid, "iem.backup", path=out_path + ".bak")
        except Exception as be:
            log_json("warning", rid, "iem.backup.error", error=str(be))

        IEMIndex.save_json(out_path, iem.to_json())
        IEM = iem

        log_json("info", rid, "iem.build", dim=iem.dim, fields=len(iem.fields))
        return BuildIEMResponse(**ok(rid, {"dim": iem.dim, "fieldCount": len(iem.fields), "savedTo": out_path}))
    except Exception as e:
        log_json("error", rid, "iem.build.error", error=str(e))
        return err(500, rid, str(e))


@app.post("/intent/encode", response_model=IntentEncoding)
def intent_encode(req: IntentRequest = Body(...), request: Request = None):
    """
    Encodes the intent. If an in-memory IEM is present, returns vector aligned
    to IEM vocab; else builds a free vocab (still deterministic).
    """
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    if IEM is not None:
        vec = encode_intent_to_vocab(req.intent, IEM.vocab)
        log_json("info", rid, "intent.encode", dim=len(vec))
        return IntentEncoding(version="intent/0.2", dim=len(IEM.vocab), vec=vec, vocab=IEM.vocab, requestId=rid)
    else:
        vec, vocab = encode_intent_to_vocab(req.intent, None, return_vocab=True)
        log_json("info", rid, "intent.encode", dim=len(vec))
        return IntentEncoding(version="intent/0.2", dim=len(vocab), vec=vec, vocab=vocab, requestId=rid)


@app.post("/intent/encode_nl", response_model=IntentEncodingNL)
def intent_encode_nl(req: NLRequest = Body(...), request: Request = None):
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    if IEM is None:
        return err(400, rid, "IEM not loaded. Call /iem/build first.")
    try:
        synonyms = load_synonyms()
        tenant = getattr(request.state, "tenant_id", None)
        time_cfg = load_time_cfg(tenant)
        out = encode_intent_nl(req.text, IEM, synonyms, topK=(req.topK or 8), time_cfg=time_cfg)
        log_json("info", rid, "intent.encode_nl", dim=out["dim"])
        return IntentEncodingNL(
            version=out["version"],
            dim=out["dim"],
            vec=out["vec"],
            vocab=IEM.vocab,
            debug={
                "topAliasHits": out["topAliasHits"],
                "blend": out["blend"],
                "text": req.text,
                **out.get("debug", {}),
            },
            requestId=rid
        )
    except Exception as e:
        log_json("error", rid, "intent.encode_nl.error", error=str(e))
        return err(500, rid, str(e))


@app.post("/synapse/match", response_model=MatchResponse)
def synapse_match(req: MatchRequest = Body(...), request: Request = None):
    """
    Runs matcher over current IEM using:
      - provided intent object (encoded to IEM vocab), or
      - provided raw vec aligned to IEM vocab (same length).
    """
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    if IEM is None:
        return err(400, rid, "IEM not loaded. Call /iem/build first.")

    if req.intentVec is not None:
        intent_vec = req.intentVec
        if len(intent_vec) != len(IEM.vocab):
            return err(400, rid, "intentVec dimension does not match current IEM vocab")
    else:
        intent_vec = encode_intent_to_vocab(req.intent or {}, IEM.vocab)

    out = match_candidates(
        iem=IEM,
        intent_vec=intent_vec,
        top_k_fields=req.topKFields or 8,
        top_k_entities=req.topKEntities or 5,
        role_alpha=req.roleAlpha,
        intent_obj=(req.intent or {}),
        alias_alpha=req.aliasAlpha,
        shape_alpha=req.shapeAlpha,
        shape_beta=req.shapeBeta,
        metric_alpha=req.metricAlpha,
    )
    out.debug["tenantId"] = getattr(request.state, "tenant_id", None)
    experiment = getattr(request.state, "experiment", "")
    source = "tenant" if experiment == "baseline-only" else "active"
    path_scoring = synapse._CONFIG.get("pathScoring", {})
    if isinstance(path_scoring, dict):
        path_scoring = {**path_scoring, "_source": source}
    out.debug["pathScoring"] = path_scoring
    out.debug["requestId"] = rid
    try:
        do_shadow = TRAINER_SHADOW_ONLY or (getattr(request.state, "experiment", "") == "train-shadow")
        if do_shadow:
            _apply_baseline_cfg()
            base = match_candidates(IEM, intent_vec, top_k_fields=req.topKFields or 8, top_k_entities=req.topKEntities or 5, intent_obj=(req.intent or {}))
            out.debug["shadow"] = {
                "mode": "baseline-vs-active",
                "baseline": {
                    "topFields": [f"{sf.entity}.{sf.name}" for sf in base.topFields[:6]],
                    "topEntities": [e.entity for e in base.topEntities[:5]]
                },
                "active": {
                    "topFields": [f"{sf.entity}.{sf.name}" for sf in out.topFields[:6]],
                    "topEntities": [e.entity for e in out.topEntities[:5]]
                }
            }
    except Exception as _e:
        out.debug["shadowError"] = str(_e)
    log_json("info", rid, "synapse.match", topFields=len(out.topFields))
    return out


@app.get("/iem/verify", response_model=VerifyIEMResponse)
def iem_verify(request: Request = None):
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    if IEM is None:
        return err(400, rid, "IEM not loaded")

    role_boost_stats: List[RoleBoostStat] = []
    for f in IEM.fields:
        top_role = max(f.role.items(), key=lambda kv: kv[1])[0] if f.role else None
        score = f.roleBoost if f.roleBoost is not None else compute_role_boost(f)
        role_boost_stats.append(RoleBoostStat(
            entity=f.entity, field=f.name, roleTop=top_role, boostScore=round(float(score), 4)
        ))

    drift_alerts: List[DriftAlert] = []
    prev_path = (IEM_PATH + ".bak")
    if os.path.exists(prev_path):
        try:
            prev = IEMIndex.load(prev_path)
            prev_map = {(p.entity, p.name): p for p in prev.fields}
            for f in IEM.fields:
                p = prev_map.get((f.entity, f.name))
                if p:
                    alias_diff, vec_diff = compute_alias_stability(f.aliases, p.aliases, f.vec, p.vec)
                    drift_alerts.append(DriftAlert(
                        field=f"{f.entity}.{f.name}",
                        aliasChange=round(alias_diff, 4),
                        vecDrift=round(vec_diff, 4),
                        warning="high drift" if (alias_diff > 0.5 or vec_diff > 0.5) else None
                    ))
        except Exception as e:
            log_json("warning", rid, "iem.verify.drift.error", error=str(e))

    clusters: List[CQ] = []
    for role in ["id", "timestamp", "money", "geo", "category", "text", "quantity"]:
        score = cluster_quality(IEM.fields, role)
        if score is not None:
            count = sum(1 for f in IEM.fields if role in f.role)
            clusters.append(CQ(role=role, avgSim=round(float(score), 4), count=count))

    warnings = [f"Low cluster cohesion for role {c.role}" for c in clusters if c.avgSim < 0.5]

    log_json("info", rid, "iem.verify", rbCount=len(role_boost_stats), driftCount=len(drift_alerts))
    return VerifyIEMResponse(
        ok=True,
        roleBoost=role_boost_stats,
        drift=drift_alerts,
        clusters=clusters,
        warnings=warnings,
        requestId=rid
    )


def _role_of(field) -> str:
    if not field.role: return "unknown"
    return max(field.role.items(), key=lambda kv: kv[1])[0]

def _is_money_op(metric: Dict[str, Any] | None) -> bool:
    if not metric: return False
    op = (metric or {}).get("op","").lower()
    return op in {"sum","avg","mean","median"}

def _apply_preferences(
    cands: List[Dict[str, Any]],
    prefer_roles: Optional[List[str]] = None,
    prefer_targets: Optional[List[str]] = None,
    prefer_targets_boost: Optional[Dict[str, float]] = None,
    alias_targets: Optional[List[str]] = None,
    ask: Optional[str] = None,
    money_op: bool = False,
    fill_cfg: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Apply weighted preferences to candidate fields.
    Returns list of dicts with updated scores.
    """
    fill_cfg = fill_cfg or {}
    role_bonus = float(fill_cfg.get("preferBonus", 0.08))
    target_bonus = 0.12
    alias_bonus = 0.30
    granularity_bonus_city = 0.12
    money_penalty = float(fill_cfg.get("moneyPivotPenalty", 0.08))
    unknown_penalty = float(fill_cfg.get("unknownPenalty", 0.1))
    
    idx = {(c["entity"], c["name"]): i for i, c in enumerate(cands)}
    
    def bump(field: str, delta: float):
        """Apply delta boost to a field if it exists."""
        if "." in field:
            ent, name = field.split(".", 1)
            j = idx.get((ent, name))
            if j is not None:
                cands[j]["score"] += delta
    
    if prefer_roles:
        pr = set(prefer_roles)
        for c in cands:
            if c.get("roleTop") in pr:
                c["score"] += role_bonus
    
    if prefer_targets:
        for t in prefer_targets:
            bump(t, target_bonus)
    
    if prefer_targets_boost:
        for t, delta in prefer_targets_boost.items():
            bump(t, float(delta))
    
    if alias_targets:
        for t in alias_targets:
            if "." in t and t.count(".") == 1:
                bump(t, alias_bonus)
    
    if ask == "top_k":
        bump("orders.shipping_city", granularity_bonus_city)
    
    if money_op:
        for c in cands:
            if c.get("roleTop") == "money":
                c["score"] -= money_penalty
    
    for c in cands:
        if c.get("roleTop") == "unknown":
            c["score"] -= unknown_penalty
    
    cands.sort(key=lambda x: x["score"], reverse=True)
    return cands

def _explain_candidate_scores(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate explain output for candidate scores."""
    return [
        {
            "field": f"{c['entity']}.{c['name']}",
            "role": c.get("roleTop", "unknown"),
            "score": round(c["score"], 6)
        }
        for c in cands
    ]

@app.get("/iem/show")
def iem_show(request: Request = None):
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    if IEM is None:
        return err(400, rid, "IEM not loaded")
    ents = defaultdict(list)
    for f in IEM.fields:
        ents[f.entity].append({
            "name": f.name,
            "roleTop": max(f.role.items(), key=lambda kv: kv[1])[0] if f.role else "unknown",
            "roles": f.role,
            "aliases": f.aliases
        })
    out = []
    for e, fields in ents.items():
        out.append({"entity": e, "fields": fields})
    return ok(rid, {"dim": IEM.dim, "entities": out})

@app.post("/synapse/fill", response_model=FillResponse)
def synapse_fill(req: FillRequest = Body(...), request: Request = None):
    """Slot filling + conflict resolve + AlignPlus™."""
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    if IEM is None:
        return err(400, rid, "IEM not loaded. Call /iem/build first.")
    
    try:
        fill_cfg = dict(synapse._CONFIG["fill"])
        if req.fillOverrides:
            for k, v in (req.fillOverrides or {}).items():
                if k in fill_cfg and v is not None:
                    fill_cfg[k] = v

        top_k = max(1, req.topKTargets or 8)
        match_out = match_candidates(
            iem=IEM,
            intent_vec=encode_intent_to_vocab(req.intent or {}, IEM.vocab),
            top_k_fields=max(64, top_k * 4),
            top_k_entities=10,
            role_alpha=(req.intent.get("roleAlpha") if isinstance(req.intent, dict) else None),
            intent_obj=req.intent or {},
        )

        money_op = _is_money_op(req.intent.get("metric"))

        raw_fields = match_out.topFields[:]
        
        cands = [{"entity": sf.entity, "name": sf.name, "score": float(sf.score), "roleTop": sf.roleTop or _role_of(sf)} for sf in raw_fields]
        
        adjusted_dicts = _apply_preferences(
            cands,
            prefer_roles=req.preferRoles,
            prefer_targets=req.preferTargets,
            prefer_targets_boost=req.preferTargetsBoost,
            alias_targets=req.aliasTargets,
            ask=req.intent.get("ask") if isinstance(req.intent, dict) else None,
            money_op=money_op,
            fill_cfg=fill_cfg
        )
        
        adjusted = [ScoredField(entity=c["entity"], name=c["name"], score=c["score"], roleTop=c["roleTop"]) for c in adjusted_dicts]
        
        adjusted.sort(key=lambda x: x.score, reverse=True)

        max_unknown = int(fill_cfg["maxUnknownInTopK"])
        picked, unknown_count = [], 0
        for cand in adjusted:
            if len(picked) >= top_k: break
            if cand.roleTop == "unknown":
                if unknown_count >= max_unknown:
                    continue
                unknown_count += 1
            picked.append(cand)

        if len(picked) < top_k:
            for cand in adjusted:
                if len(picked) >= top_k: break
                if cand in picked: continue
                if cand.roleTop != "unknown":
                    picked.append(cand)

        pivot = picked[0] if picked else (adjusted[0] if adjusted else None)
        intent_filled = dict(req.intent)
        if pivot:
            intent_filled["target"] = f"{pivot.entity}.{pivot.name}"

        intent_vec = encode_intent_to_vocab(intent_filled, IEM.vocab)
        conflicts_resolved, upgraded_target = resolve_conflicts(
            iem=IEM,
            intent_vec=intent_vec,
            intent_obj=intent_filled,
            raw_fields=raw_fields,
            pivot=pivot,
            match_debug=match_out.debug
        )

        if upgraded_target:
            intent_filled["target"] = upgraded_target

        abase = 0.0
        if pivot:
            for sf in raw_fields:
                if sf.entity == pivot.entity and sf.name == pivot.name:
                    abase = float(sf.score); break
        ap = compute_alignplus(abase, intent_filled, tokens=[], mapped_terms=set(), shaping_cfg={"alignPlus": synapse._CONFIG.get("alignPlus", {"osrZeroWhenNoText": True})})
        aplus = ap["Aplus"]
        coverage = ap["Coverage"]
        off_schema_rate = ap["OffSchemaRate"]

        ents = [ScoredEntity(entity=e.entity, score=float(e.score)) for e in match_out.topEntities]

        debug = {
            "topAliasHits": [],
            "tokens": [],
            "mappedTerms": [],
            "rolePrior": match_out.debug.get("rolePrior", {}),
            "matchDebug": match_out.debug,
            "fillConfigUsed": fill_cfg,
            "preferRolesUsed": list(req.preferRoles or []),
            "preferTargetsUsed": req.preferTargets or [],
            "preferTargetsBoostUsed": req.preferTargetsBoost or {},
            "aliasTargetsUsed": req.aliasTargets or [],
        }
        
        if req.debugExplain:
            debug["targetExplain"] = _explain_candidate_scores(adjusted_dicts)

        resp = FillResponse(
            ok=True,
            intentFilled=intent_filled,
            targetCandidates=picked,
            entityCandidates=ents,
            conflicts=[ConflictNote(
                slot=c["slot"],
                candidates=c["candidates"],
                resolution=c.get("resolution"),
                why=c.get("why", {}),
                scores=c.get("scores")
            ) for c in conflicts_resolved],
            alignPlus=AlignPlus(Abase=ap["Abase"], Coverage=ap["Coverage"], OffSchemaRate=ap["OffSchemaRate"], Aplus=ap["Aplus"]),
            debug=debug,
            requestId=rid
        )
        try:
            do_shadow = TRAINER_SHADOW_ONLY or (getattr(request.state, "experiment", "") == "train-shadow")
            if do_shadow:
                _apply_baseline_cfg()
                base_match = match_candidates(
                    iem=IEM,
                    intent_vec=encode_intent_to_vocab(req.intent or {}, IEM.vocab),
                    top_k_fields=max(64, (req.topKTargets or 8) * 4),
                    top_k_entities=10,
                    intent_obj=req.intent or {},
                )
                base_top = base_match.topFields[0] if base_match.topFields else None
                resp.debug["shadow"] = {
                    "mode": "baseline-vs-active",
                    "baselineTop": f"{base_top.entity}.{base_top.name}" if base_top else None,
                    "activeTop": resp.intentFilled.get("target"),
                }
        except Exception as _e:
            resp.debug["shadowError"] = str(_e)
        return resp
    except Exception as e:
        log_json("error", rid, "synapse.fill.error", error=str(e))
        return err(500, rid, str(e))


@app.post("/synapse/paths", response_model=PathsResponse)
def synapse_paths(req: PathsRequest = Body(...), request: Request = None):
    """Join path inference with PathScore™."""
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    if IEM is None:
        return err(400, rid, "IEM not loaded. Call /iem/build first.")
    try:
        startE = None; goalE = None; intent_vec = None

        if req.intent:
            intent_vec = encode_intent_to_vocab(req.intent, IEM.vocab)
            metric_op = ((req.intent.get("metric") or {}).get("op"))
            target = req.intent.get("target")
            startE = None
            if metric_op in {"sum","avg","mean","median"}:
                m = match_candidates(IEM, intent_vec, top_k_fields=8, top_k_entities=5, intent_obj=req.intent)
                money_entity = next((sf.entity for sf in m.topFields if sf.roleTop == "money"), None)
                startE = money_entity or (m.topEntities[0].entity if m.topEntities else None)
            if not startE:
                startE = req.intent.get("entity")
            if target:
                goalE = target.split(".")[0] if "." in target else target

        if req.start:
            se, _ = _parse_entity_field(req.start)
            if se: startE = se
        if req.goal:
            ge, _ = _parse_entity_field(req.goal)
            if ge: goalE = ge

        if not startE or not goalE:
            return PathsResponse(ok=True, paths=[], debug={"warn": "start or goal entity missing", "requestId": rid})

        adj = _build_join_graph(IEM)
        raw_paths = _bfs_paths(adj, start=startE, goal=goalE, max_hops=req.maxHops)

        ps_cfg = dict(synapse._CONFIG.get("pathScoring", {}))
        warns = []
        
        if req.pathScoringOverrides:
            for k, v in (req.pathScoringOverrides or {}).items():
                if k in ps_cfg and v is not None:
                    clamped = clamp(f"pathScoring.{k}", float(v))
                    if clamped != float(v):
                        warns.append(f"clamped {k} from {v} to {clamped}")
                    ps_cfg[k] = clamped
        
        raw = []
        for p in raw_paths:
            s = _path_score(IEM, intent_vec or [0.0]*IEM.dim, p, cfg=ps_cfg)
            edges = [PathEdge(srcEntity=a, srcField=b, dstEntity=c, dstField=d, why=e) for (a,b,c,d,e) in p]
            raw.append((s, edges))
        
        raw.sort(key=lambda t: t[0], reverse=True)
        max_s = raw[0][0] if raw else 1.0
        
        # Optional LTR re-ranking:
        ltr_enabled = LTR_DEFAULT or (request.headers.get("x-ltr","").lower() in {"1","true"}) or (request.query_params.get("ltr","") in {"1","true"})
        want_feats = (request.headers.get("x-features","").lower() in {"1","true"}) or (request.query_params.get("features","") in {"1","true"})
        ltr_debug = None
        deg_index = None
        if ltr_enabled and raw:
            deg_index = build_degree_index(IEM)
            before_top = raw[0][0]
            # Convert PathEdge objects back to raw edge tuples for LTR
            # Keep mapping to restore PathEdge objects after re-ranking
            edge_map = {}
            raw_for_ltr = []
            for s, path_edges in raw:
                raw_edges = [(e.srcEntity, e.srcField, e.dstEntity, e.dstField, e.why or "") for e in path_edges]
                edge_key = tuple(raw_edges)
                edge_map[edge_key] = path_edges
                raw_for_ltr.append((s, raw_edges))
            
            raw_reranked = ltr_rerank_paths(IEM, raw_for_ltr, ps_cfg, degree_index=deg_index)
            # Convert back to (score, PathEdge_list) format using the mapping
            raw = [(s, edge_map[tuple(edges)]) for s, edges in raw_reranked]
            after_top = raw[0][0] if raw else 0.0
            ltr_debug = {
                "enabled": True,
                "beforeTop": before_top,
                "afterTop": after_top,
                "degIndexSize": len(deg_index)
            }
        
        # Build degree index if needed (for features, edge breakdowns, or path components)
        if not deg_index:
            deg_index = build_degree_index(IEM)
        
        # helpers to compute path-level components like _path_score
        def _path_components(iem, intent_vec, edges, ps_cfg, deg_idx):
            # handle zero-hop (start==goal) exactly like _path_score
            if not edges:
                return {
                    "avgCos": 0.0,
                    "avgPrior": 0.0,
                    "rawBlend": 0.0,
                    "lengthPrior": 1.0,
                    "final": 1.0
                }
            # replicate _path_score internals to expose avgCos/avgPrior/raw/length/final
            id_prior = float(ps_cfg.get("idPrior", 0.10))
            fk_bonus = float(ps_cfg.get("fkBonus", 0.05))
            len_base = float(ps_cfg.get("lengthPriorBase", 0.92))
            w_cos    = float(ps_cfg.get("cosineWeight", 0.75))
            pk_fk_bonus = float(ps_cfg.get("pkToFkBonus", 0.00))
            fk_pk_bonus = float(ps_cfg.get("fkToPkBonus", 0.00))
            hub_pen  = float(ps_cfg.get("hubPenalty", 0.00))
            hub_th   = int(ps_cfg.get("hubDegreeThreshold", 8))
            fan_pen  = float(ps_cfg.get("fanoutPenalty", 0.00))
            
            fmap = {(f.entity, f.name): f for f in iem.fields}
            cos_terms, prior_terms = [], []
            for (se, sf, de, df, why) in edges:
                a = fmap.get((se, sf)); b = fmap.get((de, df))
                ca = max(0.0, _cosine(intent_vec, getattr(a, "vec", None))) if a else 0.0
                cb = max(0.0, _cosine(intent_vec, getattr(b, "vec", None))) if b else 0.0
                cos_terms.append((ca + cb) * 0.5)
                p = 0.0
                if sf.endswith("_id") or df.endswith("_id"):
                    p += id_prior
                if why in {"fk", "fk_rev"}:
                    p += fk_bonus
                    p += pk_fk_bonus if why == "fk" else fk_pk_bonus
                ddeg = max(0, deg_idx.get(de, 0))
                if ddeg >= hub_th:
                    p -= hub_pen
                if fan_pen:
                    p -= fan_pen * max(0, ddeg - 1)
                prior_terms.append(p)
            avg_cos = sum(cos_terms)/len(cos_terms) if cos_terms else 0.0
            avg_pri = sum(prior_terms)/len(prior_terms) if prior_terms else 0.0
            raw = (w_cos * avg_cos) + ((1.0 - w_cos) * avg_pri)
            hops = len(edges)
            length_prior = (len_base ** max(0, hops - 1))
            final = raw * length_prior
            return {
                "avgCos": float(round(avg_cos, 6)),
                "avgPrior": float(round(avg_pri, 6)),
                "rawBlend": float(round(raw, 6)),
                "lengthPrior": float(round(length_prior, 6)),
                "final": float(round(final, 6))
            }
        
        out: List[PathCandidate] = []
        path_features = []
        edge_scores_by_path = []   # NEW: per-path, per-edge breakdowns
        path_components = []       # NEW: per-path component summary
        score_norm_top = []        # NEW: normalized scores for topK
        
        # avoid div-by-zero; if no paths, keep max at 1.0
        safe_max = max_s if (raw and max_s > 0.0) else 1.0
        
        for s, edges in raw[:req.topK]:
            item = PathCandidate(path=edges, score=s, hops=len(edges))
            out.append(item)
            
            # optional features
            if want_feats:
                raw_edges = [(e.srcEntity, e.srcField, e.dstEntity, e.dstField, e.why or "") for e in edges]
                feats = _path_feats(raw_edges, deg_index)
                path_features.append(feats)
            
            # NEW: per-edge breakdown for this path
            raw_edges_for_dbg = [(e.srcEntity, e.srcField, e.dstEntity, e.dstField, e.why or "") for e in edges]
            edge_breaks = _path_edge_breakdowns(IEM, (intent_vec or [0.0]*IEM.dim), raw_edges_for_dbg, ps_cfg, deg_index)
            edge_scores_by_path.append(edge_breaks)
            
            # NEW: per-path component summary
            comp = _path_components(IEM, (intent_vec or [0.0]*IEM.dim), raw_edges_for_dbg, ps_cfg, deg_index)
            path_components.append(comp)
            
            # NEW: normalized score for UI sliders (0..1)
            score_norm_top.append(float(s / safe_max))

        experiment = getattr(request.state, "experiment", "")
        source = "tenant" if experiment == "baseline-only" else "active"
        ps_cfg_with_source = {**ps_cfg, "_source": source}

        debug_dict = {
            "start": startE, 
            "goal": goalE, 
            "maxHops": req.maxHops, 
            "found": len(raw),
            "pathScoringUsed": ps_cfg_with_source,
            "scoreMax": max_s,
            "scoreNormHint": "score / scoreMax",
            "scoreNormTop": score_norm_top,             # NEW
            "edgeScoresByPath": edge_scores_by_path,    # NEW (aligned to returned paths)
            "scoreComponentsByPath": path_components,   # NEW
            "warn": warns if warns else None,
            "requestId": rid,
            "ltr": ltr_debug or {"enabled": False}
        }
        if want_feats and path_features:
            debug_dict["pathFeatures"] = path_features
        
        return PathsResponse(
            ok=True, 
            paths=out, 
            debug=debug_dict
        )
    except Exception as e:
        log_json("error", rid, "synapse.paths.error", error=str(e))
        return err(500, rid, str(e))


@app.post("/synapse/explain", response_model=ExplainResponse)
def synapse_explain(req: ExplainRequest = Body(...), request: Request = None):
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    if IEM is None:
        return err(400, rid, "IEM not loaded. Call /iem/build first.")
    try:
        fill_req = FillRequest.model_validate({
            "intent": req.intent,
            "topKTargets": req.topKTargets or 6,
            "preferRoles": req.preferRoles or [],
            "fillOverrides": req.fillOverrides or {}
        })
        f = synapse_fill(fill_req, request)
        if isinstance(f, JSONResponse):
            return f
        fobj: FillResponse = f

        intent_vec = encode_intent_to_vocab(req.intent or {}, IEM.vocab)
        explains: List = []
        for c in fobj.conflicts:
            scored: List = []
            for tgt in c.candidates[:12]:
                ctx = score_target_context(IEM, intent_vec, tgt, intent_obj=req.intent)
                final_score = 0.5*ctx["PathScore"] + 0.35*ctx["CosineContext"] + 0.15*ctx["RoleCoherence"]
                scored.append(CandidateScore(
                    target=tgt,
                    score=float(final_score),
                    PathScore=float(ctx["PathScore"]),
                    CosineContext=float(ctx["CosineContext"]),
                    RoleCoherence=float(ctx["RoleCoherence"]),
                ))
            scored.sort(key=lambda s: s.score, reverse=True)
            explains.append(ExplainItem(slot=c.slot, resolution=c.resolution or scored[0].target, top=scored[:8]))

        resp = ExplainResponse(
            intentFilled=fobj.intentFilled,
            targetChosen=fobj.intentFilled.get("target"),
            explains=explains,
            alignPlus=fobj.alignPlus,
            debug={},
            requestId=rid
        )
        try:
            log_decision(
                event="synapse.explain",
                rid=rid,
                payload={
                    "intent": req.intent,
                    "targetChosen": resp.targetChosen,
                    "explains": [e.model_dump() for e in explains]
                },
                path=synapse._CONFIG["logging"]["path"],
                enabled=synapse._CONFIG["logging"]["decisionsEnabled"],
            )
        except Exception as _e:
            log_json("warning", rid, "decision.log.error", error=str(_e))
        return resp
    except Exception as e:
        log_json("error", rid, "synapse.explain.error", error=str(e))
        return err(500, rid, str(e))


@app.get("/examples/coverage")
def get_coverage():
    """Return cached coverage from last /generate call."""
    if COVERAGE_CACHE is None:
        return JSONResponse({"entities": [], "ok": False, "error": "no coverage cached yet. call /generate first."}, status_code=404)
    return {"entities": COVERAGE_CACHE}


@app.post("/generate", response_model=GenerateExampleResponse)
def generate(req: GenerateRequest = Body(...), request: Request = None):
    rid = getattr(request.state, "request_id", str(uuid.uuid4()))
    if IEM is None:
        return err(400, rid, "IEM not loaded. Call /iem/build first.")
    try:
        global COVERAGE_CACHE
        items_raw, cover = generate_all(IEM, per_entity=req.perEntity, strict=req.strict)
        COVERAGE_CACHE = cover
        written = 0
        cov_path = None
        target_dir = None
        if req.writeFiles:
            target_dir = os.path.abspath(req.outDir)
            if not (target_dir + os.sep).startswith(OUT_BASE + os.sep):
                return err(400, rid, f"outDir must be inside NEURONS_OUT_BASE ({OUT_BASE})")
            written = write_examples(items_raw, target_dir)
            if req.writeCoverage:
                cov_path = write_coverage(cover, target_dir)
        items = [GenIQL(**it) for it in items_raw]
        coverage = [CoverageStat(**c) for c in cover]
        log_json("info", rid, "generate.examples", total=len(items), written=written, outDir=(target_dir if req.writeFiles else None))
        return GenerateExampleResponse(**ok(rid, {"total": len(items), "written": written, "outDir": (target_dir if req.writeFiles else None), "coverageFile": cov_path, "items": items, "coverage": coverage}))
    except Exception as e:
        log_json("error", rid, "generate.error", error=str(e))
        return err(500, rid, str(e))