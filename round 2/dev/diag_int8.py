"""Isolate int8 loss: (emb) first stage with CPU-int8 vs GPU-fp16 arctic; (ce) int8 CE on fixed candidates."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "4"
import psutil
which = sys.argv[1]
fname = sys.argv[2] if len(sys.argv) > 2 else "model_int8.onnx"
psutil.Process().cpu_affinity([0, 2, 4, 6] if which == "emb" else [1, 3, 5, 7])
import numpy as np
from exp import *
from run_rerank import first_stage
import bench_onnx as bo

ds = "qasper_all"
c, q, rel = load(ds)
B = bm25_matrix(ds, 1.2, 0.75)
if which == "emb":
    from tokenizers import Tokenizer
    A = "onnx_models/snowflake-arctic-embed-m-v1.5"

    def enc(texts):
        bo.docs = texts
        e, _ = bo.run_embed(A, fname, bs=8)
        return e

    def fn():
        D = enc([clean_for_model(x["document_text"]) for x in c])
        Q = enc(["Represent this sentence for searching relevant passages: " + x["query"] for x in q])
        return (Q @ D.T).astype(np.float32)

    S8 = cached(f"{ds}__arctic-m-cpu-{fname}", fn)
    S16 = np.load(CACHE / f"{ds}__arctic-m.npy")
    for nm, S in (("gpu-fp16", S16), (f"cpu-{fname}", S8)):
        F = zn(S) + 0.5 * zn(B)
        print(nm, "dense", metrics(S, rel), "hybrid", metrics(F, rel))
else:
    F, rel, c, q = first_stage(ds)
    top = np.argsort(-F, axis=1, kind="stable")[:, :8]
    s16 = np.load(CACHE / f"{ds}__rr_gte-mbert_k30.npy")[:, :8]
    from tokenizers import Tokenizer
    import onnxruntime as ort
    G = "onnx_models/gte-reranker-modernbert-base"
    tok = Tokenizer.from_file(f"{G}/tokenizer.json"); tok.enable_truncation(512, strategy="only_second"); tok.enable_padding()
    s = bo.sess(f"{G}/onnx/{fname}")

    def fn():
        out = np.zeros((len(q), 8), dtype=np.float32)
        for i in range(len(q)):
            e = tok.encode_batch([(q[i]["query"], clean_for_model(c[j]["document_text"])) for j in top[i]])
            feed = {"input_ids": np.array([x.ids for x in e], dtype=np.int64), "attention_mask": np.array([x.attention_mask for x in e], dtype=np.int64)}
            out[i] = s.run(None, feed)[0][:, 0]
        return out

    s8 = cached(f"{ds}__rr_gte-cpu-{fname}_k8", fn)

    def z(x): return (x - x.mean()) / (x.std() + 1e-9)
    for nm, ss in (("gpu-fp16", s16), (f"cpu-{fname}", s8)):
        for a in (0.0, 0.75, 1.0):
            h = np.mean([top[i, np.argmax(z(ss[i]) + a * z(F[i, top[i]]))] in rel[i] for i in range(len(q))])
            print(nm, f"K8 a{a} hit1={h:.4f}")
    print("corr per query (mean)", np.mean([np.corrcoef(s16[i], s8[i])[0, 1] for i in range(len(q))]))
