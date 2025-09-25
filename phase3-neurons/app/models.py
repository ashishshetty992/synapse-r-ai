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


class IEMDoc(BaseModel):
    version: str = "iem/0.2"
    dim: int
    vocab: List[str]
    fields: List[IEMFieldEmbedding]


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
    intent: Dict[str, Any]  # arbitrary IQL-like object


class IntentEncoding(BaseModel):
    version: str
    dim: int
    vec: List[float]
    vocab: List[str]


# ---------- REST: /synapse/match ----------
class MatchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    intent: Optional[Dict[str, Any]] = None
    intentVec: Optional[List[float]] = None  # must be aligned to current IEM vocab
    topKFields: Optional[int] = 8
    topKEntities: Optional[int] = 5
    roleAlpha: Optional[float] = 0.12  # role prior bonus weight


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