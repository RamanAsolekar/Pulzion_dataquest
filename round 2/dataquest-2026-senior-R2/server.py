import os

import solution  # noqa: F401  # import-time work (model loading) happens before the server accepts traffic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()


def _affinity_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 0


class IngestReq(BaseModel):
    document_id: str
    document_text: str


class RetrieveReq(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok", "cpus": _affinity_cpus()}


@app.post("/ingest")
def ingest(r: IngestReq):
    solution.ingest(r.document_id, r.document_text)
    return {"ok": True}


@app.post("/retrieve")
def retrieve(r: RetrieveReq):
    doc_id = solution.retrieve(r.query)
    if not isinstance(doc_id, str) or not doc_id:
        raise HTTPException(
            status_code=500,
            detail="retrieve(query) must return exactly one non-empty document_id string",
        )
    return {"document_id": doc_id}
