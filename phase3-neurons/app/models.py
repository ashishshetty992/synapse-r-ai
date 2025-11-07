from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


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
    version: str = "iem/0.3"
    dim: int
    vocab: List[str]
    fields: List[IEMFieldEmbedding]
    roleCentroids: Optional[Dict[str, List[float]]] = None
    entities: Optional[List[IEMEntityEmbedding]] = None
    joins: Optional[List[JoinEdge]] = None


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
    requestId: Optional[str] = None


class IntentRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    intent: Dict[str, Any]


class IntentEncoding(BaseModel):
    version: str
    dim: int
    vec: List[float]
    vocab: List[str]
    requestId: Optional[str] = None


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
    field: str
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
    saliencePreview: Optional[List[Dict[str, Any]]] = None
    rolePriorVector: Optional[Dict[str, float]] = None
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
    requestId: Optional[str] = None


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
    pathScoringOverrides: Optional[Dict[str, Any]] = None


class PathsResponse(BaseModel):
    ok: bool = True
    paths: List[PathCandidate]
    debug: Dict[str, Any] = {}
    requestId: Optional[str] = None


class FillRequest(BaseModel):
    intent: Dict[str, Any]
    topKTargets: Optional[int] = 8
    preferRoles: Optional[List[str]] = None
    preferTargets: Optional[List[str]] = None
    preferTargetsBoost: Optional[Dict[str, float]] = None
    aliasTargets: Optional[List[str]] = None
    debugExplain: Optional[bool] = False
    fillOverrides: Optional[Dict[str, Any]] = None
    preferTargetEntity: Optional[bool] = True
    text: Optional[str] = None
    topKEntities: Optional[int] = 5
    roleAlpha: Optional[float] = None
    aliasAlpha: Optional[float] = None
    shapeAlpha: Optional[float] = None
    shapeBeta: Optional[float] = None
    metricAlpha: Optional[float] = None


class ConflictNote(BaseModel):
    slot: str
    candidates: List[str]
    resolution: Optional[str] = None
    selectedBy: Optional[str] = None
    decisionScore: Optional[float] = None
    bestNumericScore: Optional[float] = None
    why: Optional[Dict[str, Any]] = None
    scores: Optional[List[Dict[str, Any]]] = None


class AlignPlus(BaseModel):
    Abase: float
    Coverage: float
    OffSchemaRate: float
    Aplus: float


class ConflictOverride(BaseModel):
    slot: str
    prefer: Optional[str] = None
    avoid: Optional[List[str]] = None
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
    requestId: Optional[str] = None

class FillResponse(BaseModel):
    ok: bool = True
    intentFilled: Dict[str, Any]
    chosenTarget: Optional[str] = None  # Advisory: resolver's preferred target if different from user's
    targetCandidates: List[ScoredField]
    entityCandidates: List[ScoredEntity]
    conflicts: List[ConflictNote]
    alignPlus: AlignPlus
    debug: Dict[str, Any] = {}
    requestId: Optional[str] = None

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
    requestId: Optional[str] = None


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
    requestId: Optional[str] = None

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
    requestId: Optional[str] = None

class EvalOut(BaseModel):
    top1_acc: float
    mrr: float
    slot_f1: float
    aplus_mean: float
    pathscore_mean: float
    n_golden: int
    n_feedback: int
    ckpt: Optional[str]
    requestId: Optional[str] = None

class FewshotShowOut(BaseModel):
    tenant: str
    merged: Dict[str, Any]
    requestId: Optional[str] = None

class RolePriorVector(BaseModel):
    id: float = 0.0
    timestamp: float = 0.0
    money: float = 0.0
    geo: float = 0.0
    category: float = 0.0
    quantity: float = 0.0
    text: float = 0.0
    unknown: float = 0.0

class PrinciplesOut(BaseModel):
    ok: bool
    merged: Dict[str, Any]
    requestId: Optional[str] = None