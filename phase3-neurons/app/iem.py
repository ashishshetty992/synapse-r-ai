import math, json
from typing import Dict, List, Tuple, Iterable, Any, Optional
from collections import defaultdict

from .models import UEM, IEMDoc, IEMFieldEmbedding, IEMEntityEmbedding, JoinEdge
import numpy as np
import re



# -------- utilities --------
def char_ngrams(s: str, n_min=2, n_max=5) -> Iterable[str]:
    s = f"##{s.lower()}##"
    for n in range(n_min, n_max + 1):
        for i in range(len(s) - n + 1):
            yield s[i : i + n]


def _role_regexes():
    return {
        "id": re.compile(r"(^id$|_id$|.*_id$)", re.I),
        "timestamp": re.compile(r"(created_at|updated_at|.*_at$|timestamp|ts$|date$|dt$)", re.I),
        "money": re.compile(r"(amount|price|cost|revenue|total|subtotal|grand_total)", re.I),
        "geo": re.compile(r"(lat|lng|lon|longitude|latitude|zipcode|pincode|city|state|country|geo)", re.I),
        "category": re.compile(r"(type|status|segment|channel|method|category|source)", re.I),
        "text": re.compile(r"(name|desc|description|notes|comment|text|title)", re.I),
        "quantity": re.compile(r"(qty|quantity|units|count|num|number)", re.I),
    }


def role_probs(name: str) -> Dict[str, float]:
    rx = _role_regexes()
    matched = defaultdict(float)
    for r, pattern in rx.items():
        if pattern.search(name):
            matched[r] += 1.0
    if not matched:
        matched["unknown"] = 1.0
    s = sum(matched.values()) or 1.0
    return {k: v / s for k, v in matched.items()}


def l2_normalize(vec: Dict[str, float]) -> Dict[str, float]:
    n = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / n for k, v in vec.items()}


def dense_from_sparse(sparse: Dict[str, float], vocab: List[str]) -> List[float]:
    return [sparse.get(tok, 0.0) for tok in vocab]


# -------- IEMIndex --------
class IEMIndex:
    def __init__(self, dim: int, vocab: List[str], fields: List[IEMFieldEmbedding],
                 role_centroids: Optional[Dict[str, List[float]]] = None,
                 entity_vecs: Optional[List[IEMEntityEmbedding]] = None,
                 joins: Optional[List[JoinEdge]] = None):
        self.dim = dim
        self.vocab = vocab
        self.fields = fields
        self.role_centroids = role_centroids or {}
        self.entities = entity_vecs or []
        self.joins = joins or []  # <-- NEW

    def to_json(self) -> IEMDoc:
        return IEMDoc(
            dim=self.dim,
            vocab=self.vocab,
            fields=self.fields,
            roleCentroids=self.role_centroids or None,
            entities=(self.entities if self.entities else None),
            joins=(self.joins if self.joins else None),  # <-- NEW
        )

    @staticmethod
    def save_json(path: str, doc: IEMDoc):
        with open(path, "w") as f:
            json.dump(doc.model_dump(), f, indent=2)

    @staticmethod
    def load(path: str) -> "IEMIndex":
        data = json.load(open(path, "r"))
        doc = IEMDoc.model_validate(data)
        return IEMIndex(
            dim=doc.dim,
            vocab=doc.vocab,
            fields=doc.fields,
            role_centroids=(doc.roleCentroids or {}),
            entity_vecs=(doc.entities or []),
            joins=(doc.joins or []),  # <-- NEW
        )


def _collect_field_names(uem: UEM) -> List[Tuple[str, str]]:
    pairs = []
    for e in uem.entities:
        for f in e.fields:
            pairs.append((e.name, f.name))
    return pairs


def _build_vocab(uem: UEM, dim: int) -> List[str]:
    freq = defaultdict(int)
    for _, fname in _collect_field_names(uem):
        for g in char_ngrams(fname, 2, 5):
            freq[g] += 1
    vocab = [k for k, _ in sorted(freq.items(), key=lambda kv: kv[1], reverse=True)[:dim]]
    return vocab


def _sparse_from_name(name: str) -> Dict[str, float]:
    v = defaultdict(int)
    for g in char_ngrams(name, 2, 5):
        v[g] += 1
    return l2_normalize(v)


# + NEW: compute mean (l2-normalized) vector per role based on top-role assignment
def _compute_role_centroids(fields: List[IEMFieldEmbedding], roles: List[str]) -> Dict[str, List[float]]:
    centroids: Dict[str, List[float]] = {}
    for r in roles:
        vecs = []
        for f in fields:
            if not f.role:
                continue
            # use top role only to avoid dilution
            top_r = max(f.role.items(), key=lambda kv: kv[1])[0]
            if top_r == r:
                vecs.append(np.array(f.vec, dtype=float))
        if len(vecs) >= 1:
            m = np.mean(vecs, axis=0)
            n = np.linalg.norm(m) or 1.0
            centroids[r] = (m / n).tolist()
    return centroids


# --- helper to compute entity vectors ---
def _compute_entity_vectors(fields: List[IEMFieldEmbedding]) -> List[IEMEntityEmbedding]:
    """
    Simple, robust default:
      - mean of an entity's field vectors (L2-normalized per field),
      - optional light weighting by role (ids/timestamps slightly downweighted).
    """
    # role weights are conservative; tweak if needed
    role_weight = {
        "id": 0.8,
        "timestamp": 0.85,
        "money": 1.0,
        "geo": 1.0,
        "category": 1.0,
        "text": 0.95,
        "quantity": 1.0,
        "unknown": 0.9,
    }

    by_ent: dict[str, list[np.ndarray]] = defaultdict(list)
    for f in fields:
        v = np.array(f.vec, dtype=float)
        v = v / (np.linalg.norm(v) or 1.0)
        # pick top role for weighting
        top_role = max(f.role.items(), key=lambda kv: kv[1])[0] if f.role else "unknown"
        w = float(role_weight.get(top_role, 1.0))
        by_ent[f.entity].append(w * v)

    out: List[IEMEntityEmbedding] = []
    for ent, vecs in by_ent.items():
        if not vecs:
            continue
        m = np.mean(vecs, axis=0)
        m = m / (np.linalg.norm(m) or 1.0)
        out.append(IEMEntityEmbedding(name=ent, vec=m.tolist()))
    return out


def _extract_joins_from_uem(uem: UEM) -> List[JoinEdge]:
    """Extract FK-style join edges from UEM ref constraints."""
    joins: List[JoinEdge] = []
    ent_map = {e.name: e for e in uem.entities}
    for e in uem.entities:
        for f in e.fields:
            if f.ref and f.ref.entity in ent_map and f.ref.field:
                joins.append(JoinEdge(
                    srcEntity=e.name, srcField=f.name,
                    dstEntity=f.ref.entity, dstField=f.ref.field
                ))
    return joins


def build_iem_from_uem(uem: UEM, dim: int = 256) -> IEMIndex:
    vocab = _build_vocab(uem, dim)
    fields: List[IEMFieldEmbedding] = []

    for ent, fname in _collect_field_names(uem):
        aliases = [fname.replace("_", " "), fname]
        rp = role_probs(fname)
        sparse = _sparse_from_name(fname)
        dense = dense_from_sparse(sparse, vocab)
        # final l2
        n = math.sqrt(sum(x * x for x in dense)) or 1.0
        dense = [x / n for x in dense]

        # RoleBoost™ — compute and persist per-field
        rb = compute_role_boost_inline(dense, rp, lambda_r=0.25)

        fields.append(IEMFieldEmbedding(
            entity=ent, name=fname, aliases=aliases, role=rp, vec=dense, roleBoost=rb
        ))

    # + NEW: compute schema-adaptive role centroids (based on top-role grouping)
    roles = ["id", "timestamp", "money", "geo", "category", "text", "quantity"]
    role_centroids = _compute_role_centroids(fields, roles)  # + NEW

    # NEW: entity vectors
    entity_vecs = _compute_entity_vectors(fields)

    # NEW: extract join edges from UEM refs
    joins = _extract_joins_from_uem(uem)

    return IEMIndex(
        dim=len(vocab),
        vocab=vocab,
        fields=fields,
        role_centroids=role_centroids,
        entity_vecs=entity_vecs,
        joins=joins,  # <-- NEW
    )

# -------- Formula Lab: RoleBoost™, AliasStability™, ClusterQuality --------

def compute_role_boost_inline(vec: List[float], role: Dict[str, float], lambda_r: float = 0.25) -> float:
    """
    RoleBoost™ magnitude (stored per field): ||normalize(v + λ·R(r)) - v||
    We use a simple projection of soft role probabilities to vec length.
    """
    base = np.array(vec, dtype=float)
    role_vals = np.array(list(role.values()), dtype=float)
    if role_vals.size == 0:
        return 0.0
    # tile role vector to match base length
    role_proj = np.resize(role_vals / (np.linalg.norm(role_vals) or 1.0), base.shape)
    boosted = base + lambda_r * role_proj
    boosted = boosted / (np.linalg.norm(boosted) or 1.0)
    return float(np.linalg.norm(boosted - base))


def compute_role_boost(field: IEMFieldEmbedding, lambda_r: float = 0.25) -> float:
    """RoleBoost™ from a field object (useful for re-checks in /iem/verify)."""
    return compute_role_boost_inline(field.vec, field.role, lambda_r=lambda_r)


def compute_alias_stability(curr_aliases: List[str], prev_aliases: List[str],
                            curr_vec: List[float], prev_vec: List[float]) -> tuple[float, float]:
    """
    AliasStability™ components: (aliasChange, vecDrift)
      aliasChange = Jaccard distance on alias sets
      vecDrift    = cosine distance between embeddings
    """
    set_c, set_p = set(curr_aliases), set(prev_aliases)
    jaccard = 1.0 - (len(set_c & set_p) / max(1, len(set_c | set_p)))
    a, b = np.array(curr_vec, dtype=float), np.array(prev_vec, dtype=float)
    denom = (np.linalg.norm(a) or 1.0) * (np.linalg.norm(b) or 1.0)
    cos_sim = float(np.dot(a, b) / denom)
    vec_drift = 1.0 - cos_sim
    return jaccard, vec_drift


def cluster_quality(fields: List[IEMFieldEmbedding], role: str, top_only: bool = True) -> Optional[float]:
    role_vecs = []
    for f in fields:
        if top_only:
            if f.role:
                top = max(f.role.items(), key=lambda kv: kv[1])[0]
                if top == role:
                    role_vecs.append(np.array(f.vec, dtype=float))
        else:
            if role in f.role:
                role_vecs.append(np.array(f.vec, dtype=float))
    if len(role_vecs) < 2:
        return None
    sims = []
    for i in range(len(role_vecs)):
        for j in range(i + 1, len(role_vecs)):
            a, b = role_vecs[i], role_vecs[j]
            denom = (np.linalg.norm(a) or 1.0) * (np.linalg.norm(b) or 1.0)
            sims.append(float(np.dot(a, b) / denom))
    return sum(sims) / len(sims) if sims else None