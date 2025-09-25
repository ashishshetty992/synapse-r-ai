import math, json
from typing import Dict, List, Tuple, Iterable
from collections import defaultdict

from .models import UEM, IEMDoc, IEMFieldEmbedding


# -------- utilities --------
def char_ngrams(s: str, n_min=2, n_max=5) -> Iterable[str]:
    s = f"##{s.lower()}##"
    for n in range(n_min, n_max + 1):
        for i in range(len(s) - n + 1):
            yield s[i : i + n]


def _role_regexes():
    import re
    return {
        "id": re.compile(r"(^id$|_id$|.*_id$)", re.I),
        "timestamp": re.compile(r"(created_at|updated_at|.*_at$|timestamp|ts$|date$|dt$)", re.I),
        "money": re.compile(r"(amount|price|cost|revenue|total|subtotal|grand_total)", re.I),
        "geo": re.compile(r"(lat|lng|lon|longitude|latitude|zipcode|pincode|city|state|country|geo)", re.I),
        "category": re.compile(r"(type|status|segment|channel|method|category|country|city|state|source)", re.I),
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
    def __init__(self, dim: int, vocab: List[str], fields: List[IEMFieldEmbedding]):
        self.dim = dim
        self.vocab = vocab
        self.fields = fields

    def to_json(self) -> IEMDoc:
        return IEMDoc(dim=self.dim, vocab=self.vocab, fields=self.fields)

    @staticmethod
    def save_json(path: str, doc: IEMDoc):
        with open(path, "w") as f:
            json.dump(doc.model_dump(), f, indent=2)

    @staticmethod
    def load(path: str) -> "IEMIndex":
        data = json.load(open(path, "r"))
        doc = IEMDoc.model_validate(data)
        return IEMIndex(dim=doc.dim, vocab=doc.vocab, fields=doc.fields)


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
        fields.append(IEMFieldEmbedding(entity=ent, name=fname, aliases=aliases, role=rp, vec=dense))

    return IEMIndex(dim=len(vocab), vocab=vocab, fields=fields)