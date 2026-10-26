"""Experiment helpers: datasets, cached score matrices, metrics, fusion."""
import json
import re
from pathlib import Path

import numpy as np

DEV = Path(__file__).resolve().parent
DATA = DEV / "data"
CACHE = DEV / "cache"
CACHE.mkdir(exist_ok=True)
DATASETS = ["sample_hard", "qasper_all"]


def load(name):
    d = DATA / name
    corpus = [json.loads(l) for l in open(d / "corpus.jsonl", encoding="utf-8") if l.strip()]
    queries = [json.loads(l) for l in open(d / "queries.jsonl", encoding="utf-8") if l.strip()]
    qrels = json.load(open(d / "qrels.json"))
    ids = [c["document_id"] for c in corpus]
    pos = {x: i for i, x in enumerate(ids)}
    rel = [[pos[r] for r in qrels[q["query_id"]] if r in pos] for q in queries]
    return corpus, queries, rel


def metrics(S, rel):
    """S: Q x N score matrix (higher better)."""
    order = np.argsort(-S, axis=1, kind="stable")
    hit1 = np.mean([order[i, 0] in rel[i] for i in range(len(rel))])
    ranks = []
    for i in range(len(rel)):
        rset = set(rel[i])
        r = next(k for k in range(S.shape[1]) if order[i, k] in rset)
        ranks.append(r + 1)
    ranks = np.array(ranks)
    return {
        "hit1": round(float(hit1), 4),
        "r5": round(float(np.mean(ranks <= 5)), 4),
        "r20": round(float(np.mean(ranks <= 20)), 4),
        "r50": round(float(np.mean(ranks <= 50)), 4),
        "mrr": round(float(np.mean(1 / ranks)), 4),
    }


def zn(S):
    mu = S.mean(1, keepdims=True)
    sd = S.std(1, keepdims=True) + 1e-9
    return (S - mu) / sd


def rrf(*Ss, k=60, weights=None):
    out = 0
    weights = weights or [1] * len(Ss)
    for S, w in zip(Ss, weights):
        ranks = np.argsort(np.argsort(-S, axis=1, kind="stable"), axis=1)
        out = out + w / (k + ranks + 1)
    return out


def cached(name, fn):
    p = CACHE / f"{name}.npy"
    if p.exists():
        return np.load(p)
    S = fn()
    np.save(p, S)
    return S


def bm25_matrix(ds, k1=1.2, b=0.75, tokfn=None):
    from lexical import BM25, tokenize

    corpus, queries, rel = load(ds)
    tokfn = tokfn or tokenize
    bm = BM25([tokfn(c["document_text"]) for c in corpus], k1=k1, b=b)
    return np.stack([bm.scores(tokfn(q["query"], query=True)) for q in queries])


_CLEAN_RE = re.compile(r"\s*\[\]\s*|\s+\[(?=\d)|(?<=\d)\]")


def clean_for_model(t):
    t = _CLEAN_RE.sub(" ", t).replace("[", "").replace("]", "")
    return re.sub(r"\s+", " ", t).strip()


def _encode(model_id, texts, pooling="cls", maxlen=512, device="cuda", bs=32):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    m = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i : i + bs], padding=True, truncation=True, max_length=maxlen, return_tensors="pt").to(device)
            h = m(**enc).last_hidden_state.float()
            if pooling == "cls":
                e = h[:, 0]
            else:
                msk = enc["attention_mask"].unsqueeze(-1).float()
                e = (h * msk).sum(1) / msk.sum(1)
            out.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
    del m
    return np.concatenate(out)


def dense_matrix(ds, model_id, qp="", dp="", pooling="cls", maxlen=512, device="cuda"):
    corpus, queries, rel = load(ds)
    D = _encode(model_id, [dp + clean_for_model(c["document_text"]) for c in corpus], pooling, maxlen, device)
    Q = _encode(model_id, [qp + q["query"] for q in queries], pooling, maxlen, device)
    return (Q @ D.T).astype(np.float32)
