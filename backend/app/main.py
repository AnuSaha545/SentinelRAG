from fastapi import FastAPI

from app.api.routes.audit import router as audit_router
from app.api.routes.documents import router as documents_router


app = FastAPI(
    title="SentinelRAG",
    description="A self-healing and observable RAG system",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(
    documents_router,
    prefix="/documents",
    tags=["Documents"],
)

app.include_router(
    audit_router,
    prefix="/audit",
    tags=["Audit"],
)