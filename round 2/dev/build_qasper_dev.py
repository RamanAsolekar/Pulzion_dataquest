"""Build a QASPER-based dev set that mimics the hidden-set structure.

Hidden set (inferred from sample): ~200-word sentence-aligned chunks, 1-2 chunks
per paper, few hundred docs, few hundred specific questions, relevant = chunk(s)
containing the answer evidence.
"""
import json, re, random, sys
from pathlib import Path
import pandas as pd

OUT = Path(__file__).parent / "data" / sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "data" / "qasper_dev"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "test"
MAX_PAPERS = int(sys.argv[3]) if len(sys.argv) > 3 else 400
OUT.mkdir(parents=True, exist_ok=True)
random.seed(13)

df = pd.read_parquet(Path(__file__).parent / f"qasper_raw/qasper/{SPLIT}/0000.parquet")
SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def chunk_paper(full_text):
    """Return chunks (text, set of paragraph ids) of ~200 words, sentence aligned."""
    units = []  # (sentence, para_id)
    pid = 0
    for sec, paras in zip(full_text["section_name"], full_text["paragraphs"]):
        sec = (sec or "").split(":::")[-1].strip()
        first = True
        for p in paras:
            p = (p or "").strip()
            if not p:
                continue
            sents = SENT_RE.split(p)
            if first and sec:
                sents[0] = sec + " " + sents[0]
                first = False
            for s in sents:
                units.append((s, pid, p))
            pid += 1
    chunks, cur, cur_p, n = [], [], {}, 0
    for s, p, ptxt in units:
        w = len(s.split())
        cur.append(s)
        cur_p[p] = cur_p.get(p, 0) + w
        n += w
        if n >= 190:
            chunks.append((" ".join(cur), dict(cur_p)))
            cur, cur_p, n = [], {}, 0
    if n >= 80:
        chunks.append((" ".join(cur), dict(cur_p)))
    # paragraph word counts
    pw = {}
    for s, p, ptxt in units:
        pw[p] = pw.get(p, 0) + len(s.split())
    return chunks, pw, units


def para_index(full_text):
    m = {}
    pid = 0
    for paras in full_text["paragraphs"]:
        for p in paras:
            p = (p or "").strip()
            if not p:
                continue
            m.setdefault(p, pid)
            pid += 1
    return m


def specific(q):
    toks = q.split()
    if len(toks) >= 11:
        return True
    return any(re.search(r"[A-Z]", t) for t in toks[1:]) or any(re.search(r"\d", t) for t in toks)


docs, queries, qrels = [], [], {}
papers = list(range(len(df)))
random.shuffle(papers)
did = 0
for i in papers[:MAX_PAPERS]:
    r = df.iloc[i]
    chunks, pw, units = chunk_paper(r["full_text"])
    if len(chunks) < 4:
        continue
    pidx = para_index(r["full_text"])
    cand = []  # (question, set(chunk ids))
    qas = r["qas"]
    for q, qid, answers in zip(qas["question"], qas["question_id"], qas["answers"]):
        ev = set()
        ok = False
        for a in answers["answer"]:
            if a["unanswerable"]:
                continue
            for e in a["evidence"]:
                e = (e or "").strip()
                if not e or "FLOAT SELECTED" in e:
                    continue
                if e in pidx:
                    ev.add(pidx[e]); ok = True
        if not ok or not specific(q):
            continue
        rel = set()
        for ci, (txt, cp) in enumerate(chunks):
            for p in ev:
                if p in cp and (cp[p] >= 0.5 * pw[p] or cp[p] >= 60):
                    rel.add(ci)
        if 1 <= len(rel) <= 2:
            cand.append((q.strip(), qid, rel))
    if not cand:
        continue
    # choose up to 2 chunks per paper: most frequently relevant
    cnt = {}
    for _, _, rel in cand:
        for c in rel:
            cnt[c] = cnt.get(c, 0) + 1
    keep = sorted(cnt, key=lambda c: (-cnt[c], c))[:2]
    keep = sorted(keep)
    idmap = {}
    for c in keep:
        did += 1
        idmap[c] = f"Q{did:05d}"
        docs.append({"document_id": idmap[c], "document_text": chunks[c][0]})
    for q, qid, rel in cand:
        rr = [idmap[c] for c in rel if c in idmap]
        if rr:
            queries.append({"query_id": qid, "query": q})
            qrels[qid] = rr

with open(OUT / "corpus.jsonl", "w", encoding="utf-8") as f:
    for d in docs:
        f.write(json.dumps(d) + "\n")
with open(OUT / "queries.jsonl", "w", encoding="utf-8") as f:
    for q in queries:
        f.write(json.dumps(q) + "\n")
json.dump(qrels, open(OUT / "qrels.json", "w"), indent=1)
print(f"{OUT.name}: {len(docs)} docs, {len(queries)} queries")
