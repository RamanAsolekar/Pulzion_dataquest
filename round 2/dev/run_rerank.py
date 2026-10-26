import os
import sys
import time

os.environ["USE_TF"] = "0"
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from exp import *

K = 30
RERANKERS = {
    "minilm12": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    "bge-rr-base": "BAAI/bge-reranker-base",
    "mxbai-base": "mixedbread-ai/mxbai-rerank-base-v1",
    "gte-mbert": "Alibaba-NLP/gte-reranker-modernbert-base",
    "bge-rr-m3": "BAAI/bge-reranker-v2-m3",
    "mxbai-large": "mixedbread-ai/mxbai-rerank-large-v1",
    "bge-rr-large": "BAAI/bge-reranker-large",
}


def first_stage(ds):
    c, q, rel = load(ds)
    B = bm25_matrix(ds, 1.2, 0.75)
    S = np.load(CACHE / f"{ds}__arctic-m.npy")
    return zn(S) + 0.5 * zn(B), rel, c, q


def ce_scores(model_id, pairs, bs=16, maxlen=512):
    tok = AutoTokenizer.from_pretrained(model_id)
    m = AutoModelForSequenceClassification.from_pretrained(model_id, torch_dtype=torch.float16).cuda().eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(pairs), bs):
            a = [p[0] for p in pairs[i : i + bs]]
            b = [p[1] for p in pairs[i : i + bs]]
            enc = tok(a, b, padding=True, truncation="only_second", max_length=maxlen, return_tensors="pt").to("cuda")
            lg = m(**enc).logits.float()
            out.append((lg[:, 0] if lg.shape[1] == 1 else lg[:, 1] - lg[:, 0]).cpu().numpy())
    del m
    torch.cuda.empty_cache()
    return np.concatenate(out)


def rerank_matrix(ds, name):
    F, rel, c, q = first_stage(ds)
    top = np.argsort(-F, axis=1, kind="stable")[:, :K]

    def fn():
        pairs = [(q[i]["query"], clean_for_model(c[j]["document_text"])) for i in range(len(q)) for j in top[i]]
        t0 = time.time()
        s = ce_scores(RERANKERS[name], pairs).reshape(len(q), K)
        print(f"    [{name} scored {len(pairs)} pairs in {time.time()-t0:.1f}s on GPU]")
        return s

    s = cached(f"{ds}__rr_{name}_k{K}", fn)
    return F, top, s, rel


def evaluate(ds, name):
    F, top, s, rel = rerank_matrix(ds, name)
    Q, N = F.shape
    res = {}
    for kk in (10, 20, 30):
        R = np.full((Q, N), -1e9, dtype=np.float32)
        for i in range(Q):
            R[i, top[i, :kk]] = s[i, :kk]
        res[f"rr@{kk}"] = metrics(R, rel)["hit1"]
    # fusion with first stage (z within candidate set)
    for a in (0.1, 0.25, 0.5):
        R = np.full((Q, N), -1e9, dtype=np.float32)
        for i in range(Q):
            ss = s[i]
            fs = F[i, top[i]]
            R[i, top[i]] = (ss - ss.mean()) / (ss.std() + 1e-9) + a * (fs - fs.mean()) / (fs.std() + 1e-9)
        res[f"fuse{a}"] = metrics(R, rel)["hit1"]
    return res


if __name__ == "__main__":
    names = sys.argv[1:] or list(RERANKERS)
    for ds in DATASETS:
        F, rel, c, q = first_stage(ds)
        print(ds, "first stage", metrics(F, rel), flush=True)
        for name in names:
            try:
                print(f"  {name:12s}", evaluate(ds, name), flush=True)
            except Exception as e:
                print(f"  {name:12s} FAILED {e!r}"[:300], flush=True)
