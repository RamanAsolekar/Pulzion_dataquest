import os, sys, time, json
os.environ["OMP_NUM_THREADS"] = "4"
import psutil
psutil.Process().cpu_affinity([0, 2, 4, 6])  # 4 P-cores, one thread each
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from exp import load, clean_for_model

ds = "qasper_all"
c, q, rel = load(ds)
docs = [clean_for_model(x["document_text"]) for x in c[:128]]
qs = [x["query"] for x in q[:128]]


def sess(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def run_embed(model_dir, fname, bs=16):
    tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    tok.enable_truncation(512)
    tok.enable_padding()
    s = sess(f"{model_dir}/onnx/{fname}")
    names = [i.name for i in s.get_inputs()]
    t0 = time.time()
    outs = []
    for i in range(0, len(docs), bs):
        enc = tok.encode_batch(docs[i : i + bs])
        feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64), "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.zeros_like(feed["input_ids"])
        h = s.run(None, feed)[0]
        e = h[:, 0]
        outs.append(e / np.linalg.norm(e, axis=1, keepdims=True))
    dt = time.time() - t0
    return np.concatenate(outs), dt


def run_ce(model_dir, fname, bs=8):
    tok = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
    tok.enable_truncation(512)
    tok.enable_padding()
    s = sess(f"{model_dir}/onnx/{fname}")
    names = [i.name for i in s.get_inputs()]
    pairs = [(qs[i % len(qs)], docs[i]) for i in range(len(docs))]
    t0 = time.time()
    outs = []
    for i in range(0, len(pairs), bs):
        enc = tok.encode_batch(pairs[i : i + bs])
        feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64), "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
        if "token_type_ids" in names:
            feed["token_type_ids"] = np.array([e.type_ids for e in enc], dtype=np.int64)
        outs.append(s.run(None, feed)[0][:, 0])
    return np.concatenate(outs), time.time() - t0


if __name__ == "__main__":
    A = "onnx_models/snowflake-arctic-embed-m-v1.5"
    G = "onnx_models/gte-reranker-modernbert-base"
    e32, t32 = run_embed(A, "model.onnx")
    e8, t8 = run_embed(A, "model_pc8.onnx")
    print(f"arctic-m embed 128 docs: fp32 {t32:.1f}s ({128/t32:.1f}/s)  int8 {t8:.1f}s ({128/t8:.1f}/s)  cos(fp32,int8) mean {np.mean(np.sum(e32*e8,1)):.4f}")
    s32, t32 = run_ce(G, "model.onnx")
    s8, t8 = run_ce(G, "model_pc8.onnx")
    print(f"gte-mbert CE 128 pairs: fp32 {t32:.1f}s ({128/t32:.1f}/s)  int8 {t8:.1f}s ({128/t8:.1f}/s)  corr {np.corrcoef(s32,s8)[0,1]:.4f}")
