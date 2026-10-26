# DataQuest 2026 Senior R2: Study Plan (v2)

Deadline: **24 Oct 2026, 09:00 IST**. Plan written 23 Oct, 22:50 IST, which leaves about 10 hours.
Only the latest valid submission counts. Ship early and resubmit as the solution improves.

---

## 1. The problem

The task is retrieval only. You implement two functions in `solution.py`:

- `ingest(document_id, document_text)` is called once per document, before any query is seen.
- `retrieve(query)` returns exactly one document ID.

The score is **Hit@1**: 1 if the returned ID is one of the relevant documents for the question, otherwise 0. The final score is the mean over the hidden questions. An exception, an empty string or an unknown ID scores 0.

"Relevant" means the document **contains the answer**. Being on the same topic is not enough. The brief warns that many documents will look related without answering the question.

If two submissions tie on score, the tiebreakers are, in order: lower total execution time, then lower peak memory, then earlier submission.

## 2. Hard limits

These are now consistent between the brief (v2) and the README.

| Limit | Value |
|---|---|
| Execution time (container start + model load + all ingests + all retrieves) | **40 min total** |
| CPU | AMD Ryzen 7 7840HS (Zen 4): **4 physical cores**, SMT off, 3.8 GHz, no turbo. AVX-512 **VNNI** and BF16 are available, so int8 quantization is fast. |
| Threads | `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=4` |
| Memory | **2.5 GB**. The `/tmp` tmpfs counts against this. |
| Docker image | **2.5 GB**. Base is `python:3.11-slim` plus our requirements. |
| Build | 20 min, with network access. Downloads go in `prefetch.py`. |
| Runtime | No network, read-only root, only `/tmp` writable (1 GB), `HF_HUB_OFFLINE=1` |
| ZIP | 512 MB, at most 2.5 GB extracted, at most 10,000 files |

The harness supplies its own `server.py` and `Dockerfile`. `import solution` happens before `/health` responds, so model loading time counts toward the 40 minutes.

## 3. What the data tells us

- **Documents:** ~200-word chunks of *Atmos. Chem. Phys.* papers, at 170–220 words, about 1,000–1,400 characters, or **250–345 WordPiece tokens**. Every chunk fits a 384- or 512-token model whole, so no further chunking is needed.
- **Consecutive IDs are consecutive chunks of one paper.**
  - In 3 of the 5 sample questions, both of two adjacent chunks are relevant: D0001+D0002, D0003+D0004 and D0014+D0015.
  - Hard distractors are therefore **sibling chunks from the same paper**.
- **PDF-extraction noise:**
  - Split formulas: `PM 2 5`, `H 2 SO 4`, `CH 4`, `SO [2] 4 []`.
  - Superscript junk: `W m [][2]`.
  - Dropped "et al.": `Mu , 2007`.
  - Lost dashes: `19902011`, `AprilSeptember`.
  - Hyphenation splits: `sec ondary`, `ve hicular`.
  - Journal page footers: `Atmos. Chem. Phys., 22, 1259112606, 2022 12600 J. He : <title>`.
  - Non-content chunks such as data availability and author contributions (D0006).
- **Queries:** 17–31 tokens, phrased like **peer-reviewer comments**.
  - They contain precise anchors: "Proxy 4", "OLS instead of ODR", "14%", "NOAA to AIRS after 2011".
  - They also contain meta-clauses: "is the statement in the paper entirely accurate?" and "does the 14% difference justify the switch?".
  - Their spelling doesn't always match the documents: `PM2.5` vs `PM 2 5`, `2DS` vs `2D-S`, "changed" vs "switched".
- **The sample is useless for tuning.** The rank-bm25 baseline scores **5/5**, and every relevant document is at rank ≤ 3 of only 20. We need our own, harder development set.
- **Estimated hidden scale, inferred from the brief's baseline figures** (0.3 s ingest, 0.5 s for all queries, 46 MiB total):
  - Locally, rank-bm25 adds about 9 MiB per 1,000 docs on top of about 45–55 MiB of imports.
  - The corpus is therefore probably **a few hundred to about 3,000 chunks**, which fits 4-digit IDs like `D0001`.
  - There are probably **about 100–500 questions**.
  - This is an inference, not a guarantee, so the design must still survive about 10,000 docs and about 2,000 queries without timing out.

**What this implies:** 40 minutes on 4 AVX-512 cores is a large budget for about 2,000 short chunks. A **neural pipeline is affordable**: dense retrieval plus a cross-encoder reranker, and possibly a small LLM reranker on the last few candidates. The real constraints are a per-query time cap, because the query count is unknown, and a 2.5 GB image and RAM ceiling.

## 4. Development setup (Phase 0, most important)

1. **`dev/offline_eval.py`**
   - Imports `solution` in-process and resets its state between runs.
   - Reports Hit@1, Recall@5/10/50, MRR, the rank of the first relevant document, timings and peak RSS.
   - Runs with 4 threads pinned.
   - Also writes per-query failures for error analysis.
2. **Dev-B, the main tuning set: PeerQA** (UKP Lab, NAACL 2025, CC-BY-NC-SA, used locally only and never shipped).
   - It has reviewer-written questions over scientific papers, answered by the authors, with annotated answer evidence. This is the closest public analogue of our query style.
   - Chunk each paper into ~200-word windows. A chunk is relevant if it contains annotated evidence. Inject the same PDF noise as the hidden data.
   - Subsample to about 2,000–3,000 chunks so the scale matches.
   - Rule 3 guard: drop any paper from ACP/EGU-atmospheric venues.
3. **Dev-A, the in-domain set:** 60–100 new reviewer-style questions written over the 20 sample docs.
   - Make them deliberately hard: paraphrases, low word overlap, compound questions, and questions whose answer lies in one sibling chunk only.
   - Mix the 20 docs into the Dev-B distractor pool.
4. **Scale and timing harness:** synthetic corpora of 2k, 5k and 10k docs with 500 and 2,000 queries, timed with `OMP/ONNX threads=4` and a 4-core affinity mask.
5. **Docker check:** start Docker Desktop, which is installed but whose daemon isn't running. Then run `python local_eval.py`, which uses the official isolation flags.

**Decision rule:** keep a change only if
- it gains at least 1 point of Hit@1 on Dev-B,
- it isn't worse on Dev-A, and
- projected runtime at N=3k docs and Q=500 queries stays **≤ 20 min**, half the budget.

## 5. Approach ladder

Each rung is measured on the dev sets before the next one is built.

| # | Approach | Why | Cost at N≈2k, Q≈300 (to measure) |
|---|---|---|---|
| A0 | rank-bm25 baseline | reference number | ~1 s |
| A1 | **Lexical++**: joined chemical formulas (`pm25`, `h2so4`, `ch4`), hyphen and slash variants (`2d-s`/`2ds`), junk removal, stemming, a stopword list for question templates, `k1`/`b` tuned, fast sparse matrix BM25 (numpy/scipy) | cheap, robust, closes part of the vocabulary gap | ~1–2 s |
| A2 | **Dense bi-encoder** (ONNX int8): bge-small/base-en-v1.5, e5-base-v2, gte-base, snowflake-arctic-embed-m-v1.5/v2, nomic-embed-v1.5. Hybrid with A1 via RRF or z-score fusion. | handles paraphrases ("changed" vs "switched") | small: ~1 min embed; base: ~3–5 min |
| A3 | **Cross-encoder rerank** of the top-K hybrid candidates: ms-marco-MiniLM-L-6/L-12, bge-reranker-base, mxbai-rerank-xsmall/base, jina-reranker-v1-turbo, gte-reranker-modernbert-base. Fuse its score with the first-stage score. | the largest expected Hit@1 gain: separating "answers it" from "looks related" | MiniLM-L6 at K=30: ~0.5 s/q; base at K=20: ~1–2 s/q |
| A4 | **Corpus-structure features**: smooth scores over neighbouring chunks, group chunks into papers (adjacent IDs plus page-footer titles), and use a prior against non-content chunks (references, data availability, author contributions) | exploits the adjacency seen in the sample labels | ~0 |
| A5 | **Query cleaning**: strip reviewer meta-clauses ("is the statement accurate", "justify", "the paper"), dedicated handling for numbers and percentages | reduces noise for the dense and cross-encoder models | ~0 |
| A6 | **LLM reranker** on the top 3–5 only: Qwen3-Reranker-0.6B or bge-reranker-v2-m3 (int8) | stronger judgement of whether a chunk actually answers | ~3–6 s/q, so only if Q is small; behind a time guard |
| A7 | Stretch: fine-tune the reranker or embedder locally on the GPU using PeerQA plus synthetic questions, and ship the weights in `assets/` | adapts the models to the domain | hours, so probably out of scope |

doc2query and HyDE at runtime are ruled out. Generation on CPU is too slow for the unknown N and Q.

## 6. Engineering requirements for the final `solution.py`

- **ONNX Runtime + `tokenizers` + numpy, no torch at runtime.** This gives a smaller image (about 300 MB vs about 1.5 GB), faster startup and lower RAM, which also helps the tiebreakers.
  - Use int8 models (`*_qint8_avx512_vnni.onnx`, or quantize with optimum locally).
  - Small models ship in `assets/` (ZIP ≤ 512 MB), so the build doesn't depend on Hugging Face being reachable. Large ones download in `prefetch.py`.
- **Lazy index build.** `ingest` only normalizes, tokenizes and stores, and could optionally embed in batches as documents arrive. The first `retrieve` finishes the index.
- **Time guard.**
  - Track elapsed time since import.
  - Keep the per-query rerank cost fixed so that even 2,000 queries fit.
  - Past about 30 min, degrade to the first stage only (dense + BM25) and still answer every query.
- **Never return an invalid answer.**
  - Wrap retrieval in try/except and fall back to the best BM25 result, then the first ingested ID.
  - Handle an empty query and a corpus of size 1.
- **Determinism:** fixed thread count, stable sorting with ties broken by ID, no randomness.
- **Write nothing at runtime** except under `/tmp`. Load models with `local_files_only`.

## 7. Timeline (IST)

| Window | Work | Exit criterion |
|---|---|---|
| 23:00–00:30 | Phase 0: harness, Dev-B (PeerQA), Dev-A, Docker running | A0 numbers on both dev sets |
| 00:30–01:30 | A1 lexical++, then Docker `local_eval` | **Submission #1** (safe, better than baseline) |
| 01:30–03:30 | A2 dense model bake-off, hybrid fusion, A5 | best first stage chosen |
| 03:30–05:30 | A3 reranker bake-off, A4, time guard, ONNX packaging | **Submission #2** |
| 05:30–07:00 | A6 or other stretch, *only if* the dev gain is clear and runtime ≤ 20 min | optional **Submission #3** |
| 07:00–08:15 | Freeze: fresh template copy → unzip → `local_eval.py` (read-only, 2.5 GB, 4 cores) → `make_submission.py` | final ZIP verified |
| **≤ 08:30** | **Submit via the Google Form** (you submit it, keep the confirmation) | buffer before 09:00 |

## 8. Risks

| Risk | Mitigation |
|---|---|
| The dev sets don't match the hidden distribution | use two different dev sets and only accept gains that hold on both |
| Hidden N or Q larger than expected | time guard plus per-query cost cap; stress test at 10k docs and 2k queries |
| The build fails, e.g. HF download or pip | ship models in `assets/`, pin versions, test the Docker build from a clean copy |
| RAM or image limit exceeded | ONNX int8, no torch, measure `docker stats` peak |
| Late surprise | Submission #1 exists by about 01:30 |
