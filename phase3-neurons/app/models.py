from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# ---------- UEM (input schema) ----------
class UEMRef(BaseModel):
    entity: str
    field: str


class UEMField(BaseModel):
    name: str
    type: str
    role: Optional[str] = None
    required: Optional[bool] = None
    index: Optional[bool] = None
    ref: Optional[UEMRef] = None


class UEMEntity(BaseModel):
    name: str
    primaryKey: str
    fields: List[UEMField]


class UEM(BaseModel):
    version: str = "uem/0.1"
    database: Optional[Dict[str, Any]] = None
    entities: List[UEMEntity]


# ---------- IEM (schema embeddings) ----------
class IEMFieldEmbedding(BaseModel):
    entity: str
    name: str
    aliases: List[str]
    role: Dict[str, float]
    vec: List[float]
    roleBoost: Optional[float] = None


class IEMEntityEmbedding(BaseModel):
    name: str
    vec: List[float]


class JoinEdge(BaseModel):
    srcEntity: str
    srcField: str
    dstEntity: str
    dstField: str


class IEMDoc(BaseModel):
    version: str = "iem/0.3"  # bump minor version since schema grows
    dim: int
    vocab: List[str]
    fields: List[IEMFieldEmbedding]
    roleCentroids: Optional[Dict[str, List[float]]] = None
    entities: Optional[List[IEMEntityEmbedding]] = None
    joins: Optional[List[JoinEdge]] = None  # <-- add this


# ---------- REST: /iem/build ----------
class BuildIEMRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    uem: Optional[UEM] = None
    uemPath: Optional[str] = None
    iemPath: Optional[str] = None
    dim: Optional[int] = Field(default=256, ge=32, le=2048)


class BuildIEMResponse(BaseModel):
    ok: bool
    dim: int
    fieldCount: int
    savedTo: str


# ---------- REST: /intent/encode ----------
class IntentRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    intent: Dict[str, Any]


class IntentEncoding(BaseModel):
    version: str
    dim: int
    vec: List[float]
    vocab: List[str]


# ---------- REST: /synapse/match ----------
class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    intent: Optional[Dict[str, Any]] = None
    intentVec: Optional[List[float]] = None
    topKFields: Optional[int] = 8
    topKEntities: Optional[int] = 5
    roleAlpha: Optional[float] = None
    aliasAlpha: Optional[float] = None
    shapeAlpha: Optional[float] = None
    shapeBeta: Optional[float] = None
    metricAlpha: Optional[float] = None


class ScoredField(BaseModel):
    entity: str
    name: str
    score: float
    roleTop: Optional[str] = None


class ScoredEntity(BaseModel):
    entity: str
    score: float


class MatchResponse(BaseModel):
    ok: bool = True
    vocabDim: int
    topFields: List[ScoredField]
    topEntities: List[ScoredEntity]
    debug: Dict[str, Any] = {}

class RoleBoostStat(BaseModel):
    entity: str
    field: str
    roleTop: Optional[str]
    boostScore: float


class DriftAlert(BaseModel):
    field: str  # "entity.field"
    aliasChange: float
    vecDrift: float
    warning: Optional[str] = None


class ClusterQuality(BaseModel):
    role: str
    avgSim: float
    count: int


class VerifyIEMResponse(BaseModel):
    ok: bool
    roleBoost: List[RoleBoostStat]
    drift: List[DriftAlert]
    clusters: List[ClusterQuality]
    warnings: List[str]
    requestId: Optional[str] = None

class NLRequest(BaseModel):
    text: str
    topK: Optional[int] = 8
    tenant: Optional[str] = None

class IntentEncodingNL(BaseModel):
    version: str
    dim: int
    vec: List[float]
    vocab: List[str]
    debug: Optional[Dict[str, Any]] = None


# ---------- Neuron-3: Slot Fill + AlignPlus + PathScore ----------
class PathEdge(BaseModel):
    srcEntity: str
    srcField: str
    dstEntity: str
    dstField: str
    why: Optional[str] = None


class PathCandidate(BaseModel):
    path: List[PathEdge]
    score: float
    hops: int


class PathsRequest(BaseModel):
    intent: Optional[Dict[str, Any]] = None
    start: Optional[str] = None
    goal: Optional[str] = None
    maxHops: int = 2
    topK: int = 5
    pathScoringOverrides: Optional[Dict[str, Any]] = None  # NEW


class PathsResponse(BaseModel):
    ok: bool = True
    paths: List[PathCandidate]
    debug: Dict[str, Any] = {}


class FillRequest(BaseModel):
    intent: Dict[str, Any]
    topKTargets: Optional[int] = 8
    preferRoles: Optional[List[str]] = None
    fillOverrides: Optional[Dict[str, Any]] = None
    preferTargetEntity: Optional[bool] = True  # NEW: tie-break toward explicit target's entity
    text: Optional[str] = None  # keep for backward compat
    topKEntities: Optional[int] = 5  # keep for backward compat
    roleAlpha: Optional[float] = None
    aliasAlpha: Optional[float] = None
    shapeAlpha: Optional[float] = None
    shapeBeta: Optional[float] = None
    metricAlpha: Optional[float] = None


class ConflictNote(BaseModel):
    slot: str                              # e.g. "city"
    candidates: List[str]                  # ["orders.shipping_city","customer.city", ...]
    resolution: Optional[str] = None       # chosen one (entity.field)
    why: Optional[Dict[str, Any]] = None   # structured: {weights, factors...}
    scores: Optional[List[Dict[str, Any]]] = None  # per-candidate: [{target, score, ...why...}]


class AlignPlus(BaseModel):
    Abase: float
    Coverage: float
    OffSchemaRate: float
    Aplus: float


class ConflictOverride(BaseModel):
    slot: str                         # "category" | "timestamp" | "geo" | "text"
    prefer: Optional[str] = None      # target to prefer, e.g., "payment.method"
    avoid: Optional[List[str]] = None # targets to downrank/remove
    acceptRoles: Optional[List[str]] = None
    minPathScore: Optional[float] = None
    minCosine: Optional[float] = None

class ExplainRequest(BaseModel):
    intent: Dict[str, Any]
    topKTargets: Optional[int] = 6
    preferRoles: Optional[List[str]] = None
    conflictOverrides: Optional[List[ConflictOverride]] = None
    fillOverrides: Optional[Dict[str, Any]] = None
    aliasAlpha: Optional[float] = None
    shapeAlpha: Optional[float] = None
    shapeBeta: Optional[float] = None
    metricAlpha: Optional[float] = None

class CandidateScore(BaseModel):
    target: str
    score: float
    PathScore: float
    CosineContext: float
    RoleCoherence: float
    why: Optional[str] = None

class ExplainItem(BaseModel):
    slot: str
    resolution: str
    top: List[CandidateScore]

class ExplainResponse(BaseModel):
    ok: bool = True
    intentFilled: Dict[str, Any]
    targetChosen: str
    explains: List[ExplainItem]
    alignPlus: AlignPlus
    debug: Dict[str, Any] = {}

class FillResponse(BaseModel):
    ok: bool = True
    intentFilled: Dict[str, Any]
    targetCandidates: List[ScoredField]
    entityCandidates: List[ScoredEntity]
    conflicts: List[ConflictNote]
    alignPlus: AlignPlus
    debug: Dict[str, Any] = {}

class GenIQL(BaseModel):
    file: str
    title: str
    entity: str
    intent: Dict[str, Any]
    iql: Dict[str, Any]

class GenerateRequest(BaseModel):
    perEntity: int = 5
    outDir: str = "examples/aiql"
    writeFiles: bool = True
    strict: bool = True
    writeCoverage: bool = True
    maxHops: int = 2

class CoverageStat(BaseModel):
    entity: str
    money: bool
    timestamp: bool
    category: bool
    geo: bool
    quantity: bool

class GenerateExampleResponse(BaseModel):
    ok: bool = True
    total: int
    written: int
    outDir: Optional[str] = None
    coverageFile: Optional[str] = None
    items: List[GenIQL]
    coverage: List[CoverageStat]


# ---------- Feedback, Trainer, Eval, Fewshot ----------
class FeedbackRecordIn(BaseModel):
    tenant: str = 'default'
    intent: Dict[str, Any]
    chosenTarget: str = Field(..., description="Resolved target like 'orders.total_amount'")
    otherSlots: Dict[str, str] = Field(default_factory=dict)
    iemHash: Optional[str] = None
    latencyMs: Optional[int] = None
    sensitive: bool = False

class FeedbackRecordOut(BaseModel):
    ok: bool
    id: str

class TrainerRunIn(BaseModel):
    tenant: Optional[str] = None
    maxGrid: int = 64
    maxFewshot: int = 2000

class TrainerRunOut(BaseModel):
    ok: bool
    ckpt: Optional[str] = None
    metrics: Dict[str, Any]

class ActivateIn(BaseModel):
    checkpoint: str

class VersionsOut(BaseModel):
    versions: list[dict]

class EvalOut(BaseModel):
    top1_acc: float
    mrr: float
    slot_f1: float
    aplus_mean: float
    pathscore_mean: float
    n_golden: int
    n_feedback: int
    ckpt: Optional[str]

class FewshotShowOut(BaseModel):
    tenant: str
    merged: Dict[str, Any]