import math
from typing import List, Dict, Any
from collections import defaultdict

from .models import MatchResponse, ScoredField, ScoredEntity
from .iem import IEMIndex


def _cosine(a: List[float], b: List[float]) -> float:
    # both are already l2-normalized; dot product is cosine
    return sum(x * y for x, y in zip(a, b))


def _guess_intent_role_tokens(intent_text: str) -> Dict[str, float]:
    """
    Very light role prior from keywords inside the serialized intent text.
    """
    text = intent_text.lower()
    scores = defaultdict(float)
    tokens = {
        "money": ["revenue", "amount", "price", "total", "gmv", "sales"],
        "timestamp": ["day", "date", "week", "month", "year", "created_at"],
        "geo": ["city", "country", "state", "region"],
        "category": ["segment", "category", "status", "method", "channel"],
        "quantity": ["count", "qty", "units", "orders"],
        "text": ["name", "title"],
        "id": ["id", "order_id", "customer_id", "product_id"],
    }
    for r, kws in tokens.items():
        for k in kws:
            if k in text:
                scores[r] += 1.0
    # normalize
    s = sum(scores.values()) or 1.0
    return {k: v / s for k, v in scores.items()}


def match_candidates(
    iem: IEMIndex,
    intent_vec: List[float],
    top_k_fields: int = 8,
    top_k_entities: int = 5,
    role_alpha: float = 0.12,
) -> MatchResponse:
    """
    Scores (cosine) each field vector against the intent vector,
    with a small role prior bonus for intuitive nudging.
    """
    # serialize a pseudo-intent text from similarity hotspots for role guess (debug)
    # (We could pass the raw intent object too; here we approximate.)
    intent_signature = " ".join([str(round(x, 3)) for x in intent_vec[:32]])
    role_prior = _guess_intent_role_tokens(intent_signature)

    scored_fields: List[ScoredField] = []
    for f in iem.fields:
        base = _cosine(intent_vec, f.vec)
        # role bonus: use the top role of the field against the prior
        top_role = max(f.role.items(), key=lambda kv: kv[1])[0]
        bonus = role_prior.get(top_role, 0.0) * role_alpha
        score = base * (1.0 + bonus)
        scored_fields.append(ScoredField(entity=f.entity, name=f.name, score=score, roleTop=top_role))

    # top fields
    scored_fields.sort(key=lambda x: x.score, reverse=True)
    top_fields = scored_fields[:top_k_fields]

    # aggregate by entity (average of top-N per entity)
    per_ent = defaultdict(list)
    for sf in scored_fields[: max(64, top_k_fields * 4)]:
        per_ent[sf.entity].append(sf.score)
    ent_scores = [
        ScoredEntity(entity=e, score=sum(vs) / len(vs) if vs else 0.0) for e, vs in per_ent.items()
    ]
    ent_scores.sort(key=lambda x: x.score, reverse=True)
    top_entities = ent_scores[:top_k_entities]

    return MatchResponse(
        ok=True,
        vocabDim=iem.dim,
        topFields=top_fields,
        topEntities=top_entities,
        debug={
            "rolePrior": role_prior,
            "consideredFields": len(iem.fields),
            "vocabDim": iem.dim,
        },
    )