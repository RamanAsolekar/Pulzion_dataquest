import os, sys
os.environ["USE_TF"] = "0"
import numpy as np
from exp import *
BQ = "Represent this sentence for searching relevant passages: "
MODELS = {
    "bge-small": ("BAAI/bge-small-en-v1.5", BQ, "", "cls"),
    "bge-base": ("BAAI/bge-base-en-v1.5", BQ, "", "cls"),
    "e5-base": ("intfloat/e5-base-v2", "query: ", "passage: ", "mean"),
    "arctic-m": ("Snowflake/snowflake-arctic-embed-m-v1.5", BQ, "", "cls"),
    "bge-large": ("BAAI/bge-large-en-v1.5", BQ, "", "cls"),
}
names = sys.argv[1:] or list(MODELS)
for ds in DATASETS:
    c, q, rel = load(ds)
    B = bm25_matrix(ds, 1.2, 0.75)
    print(ds, "bm25", metrics(B, rel))
    for name in names:
        mid, qp, dp, pool = MODELS[name]
        S = cached(f"{ds}__{name}", lambda: dense_matrix(ds, mid, qp, dp, pool))
        print(f"  {name:10s}", metrics(S, rel), " rrf", metrics(rrf(S, B), rel)["hit1"],
              " z", [metrics(zn(S) + w * zn(B), rel)["hit1"] for w in (0.2, 0.35, 0.5, 0.75)], flush=True)
