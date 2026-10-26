import os, re, sys, importlib.util; os.environ["USE_TF"]="0"; os.environ["DQ_MODEL_DIR"]="/none"
import numpy as np
import exp
from exp import *
import run_rerank
spec=importlib.util.spec_from_file_location("sol","../dataquest-2026-senior-R2/solution.py"); sol=importlib.util.module_from_spec(spec); spec.loader.exec_module(sol)
exp.clean_for_model = sol._model_text
BQ="Represent this sentence for searching relevant passages: "
def z(x): return (x-x.mean())/(x.std()+1e-9)
for ds in sys.argv[1:]:
    c,q,rel=load(ds)
    tok=lambda s, query=False: sol._tokenize(s, query)
    B=bm25_matrix(ds,1.2,0.75,tok)
    S=cached(f"{ds}__arctic-m-norm", lambda: dense_matrix(ds,"Snowflake/snowflake-arctic-embed-m-v1.5",BQ,"","cls"))
    print(ds, "bm25", metrics(B,rel)["hit1"], "dense", metrics(S,rel)["hit1"], "hybrid w:", {w: metrics(zn(S)+w*zn(B),rel)["hit1"] for w in (0.3,0.5,0.75,1.0)}, flush=True)
    for w in (0.5,0.75):
        F=zn(S)+w*zn(B); top=np.argsort(-F,axis=1,kind="stable")[:,:10]
        def fn():
            pairs=[(q[i]["query"], sol._model_text(c[j]["document_text"])) for i in range(len(q)) for j in top[i]]
            return run_rerank.ce_scores("Alibaba-NLP/gte-reranker-modernbert-base", pairs).reshape(len(q),10)
        s=cached(f"{ds}__rr_gte-norm-w{w}_k10", fn)
        for gap in (1.5,2.0,None):
            row=[]
            for kk in (5,10):
                for a in (0.5,0.75,1.0):
                    h=0; nr=0
                    for i in range(len(q)):
                        if gap is not None and F[i,top[i,0]]-F[i,top[i,1]]>gap: h+=top[i,0] in rel[i]; continue
                        nr+=1; h+=top[i,np.argmax(z(s[i,:kk])+a*z(F[i,top[i,:kk]]))] in rel[i]
                    row.append(f"K{kk}a{a}:{h/len(q):.3f}")
            print(f"   w{w} gap{gap} (rerank {nr}/{len(q)}):", " ".join(row), flush=True)
