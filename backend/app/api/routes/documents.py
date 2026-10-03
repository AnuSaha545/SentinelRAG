from pathlib import Path
from time import perf_counter
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.audit import AuditLog
from app.models.document import Document, DocumentChunk
from app.services.confidence import calculate_confidence
from app.services.decision import make_decision, needs_human_review
from app.services.document_processor import process_document
from app.services.generation import generate_answer
from app.services.retrieval import (
    hybrid_retrieve_chunks,
    is_list_question,
    prepare_generation_context,
    rerank_chunks,
)
from app.services.verification import calculate_verification_features


router = APIRouter()


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@router.get("")
def list_documents(
    db: Session = Depends(get_db),
):
    documents = db.query(Document).order_by(
        Document.created_at.desc()
    ).all()

    return [
        {
            "document_id": document.id,
            "filename": document.filename,
            "created_at": document.created_at,
        }
        for document in documents
    ]


class QueryRequest(BaseModel):
    query: str
    document_id: str
    limit: int = 5

    def model_post_init(self, __context):
        if self.limit <= 0:
            raise ValueError("limit must be greater than 0")

        if not self.query.strip():
            raise ValueError("query cannot be empty")

        if not self.document_id.strip():
            raise ValueError("document_id cannot be empty")


def is_refusal(answer: str) -> int:
    refusal_phrases = [
        "I could not find the answer",
        "I don't know",
        "I cannot find",
        "not found in the document",
    ]

    answer_lower = answer.lower()

    return int(
        any(
            phrase.lower() in answer_lower
            for phrase in refusal_phrases
        )
    )


def calculate_answer_features(
    answer: str,
    results: list[dict],
) -> dict:
    similarities = [
        chunk.get("similarity", 0.0)
        for chunk in results
        if chunk.get("similarity") is not None
    ]

    rerank_scores = [
        chunk.get(
            "_confidence_rerank_score",
            chunk.get("rerank_score", 0.0),
        )
        for chunk in results
        if chunk.get("rerank_score") is not None
    ]

    vector_matches = sum(
        "vector" in chunk.get("retrieval_methods", [])
        for chunk in results
    )

    keyword_matches = sum(
        "keyword" in chunk.get("retrieval_methods", [])
        for chunk in results
    )

    hybrid_match = int(
        any(
            "vector" in chunk.get("retrieval_methods", [])
            and "keyword" in chunk.get("retrieval_methods", [])
            for chunk in results
        )
    )

    top_similarity = max(similarities) if similarities else 0.0

    max_rerank_score = (
        max(rerank_scores)
        if rerank_scores
        else 0.0
    )

    mean_rerank_score = (
        sum(rerank_scores) / len(rerank_scores)
        if rerank_scores
        else 0.0
    )

    rerank_score_spread = (
        max(rerank_scores) - min(rerank_scores)
        if rerank_scores
        else 0.0
    )

    verification_features = calculate_verification_features(
        answer=answer,
        chunks=results,
    )

    return {
        "top_similarity": top_similarity,
        "max_rerank_score": max_rerank_score,
        "mean_rerank_score": mean_rerank_score,
        "rerank_score_spread": rerank_score_spread,
        "vector_matches": vector_matches,
        "keyword_matches": keyword_matches,
        "hybrid_match": hybrid_match,
        "answer_length": len(answer),
        "is_refusal": is_refusal(answer),
        "context_answer_similarity": verification_features.get(
            "context_answer_similarity",
            0.0,
        ),
        "max_answer_chunk_similarity": verification_features.get(
            "max_answer_chunk_similarity",
            0.0,
        ),
        "max_nli_entailment": verification_features.get(
            "max_nli_entailment",
            0.0,
        ),
        "max_nli_contradiction": verification_features.get(
            "max_nli_contradiction",
            0.0,
        ),
    }


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    document_id = str(uuid4())
    file_path = UPLOAD_DIR / f"{document_id}_{file.filename}"

    content = await file.read()
    file_path.write_bytes(content)

    try:
        processed_chunks = process_document(str(file_path))
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    document = Document(
        id=document_id,
        filename=file.filename,
    )

    db.add(document)

    for chunk in processed_chunks:
        db.add(
            DocumentChunk(
                document_id=document_id,
                chunk_index=chunk["chunk_index"],
                content=chunk["text"],
                embedding=chunk["embedding"],
            )
        )

    db.commit()

    return {
        "document_id": document_id,
        "filename": file.filename,
        "chunks": len(processed_chunks),
        "status": "stored",
    }


@router.delete("/{document_id}")
def delete_document(
    document_id: str,
    db: Session = Depends(get_db),
):
    document = db.get(Document, document_id)

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    file_path = UPLOAD_DIR / f"{document_id}_{document.filename}"

    try:
        db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).delete(synchronize_session=False)

        db.delete(document)
        db.commit()

    except Exception:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="Failed to delete document from database.",
        )

    if file_path.exists():
        try:
            file_path.unlink()
        except OSError:
            raise HTTPException(
                status_code=500,
                detail=(
                    "Document was removed from the database, "
                    "but the uploaded file could not be deleted."
                ),
            )

    return {
        "document_id": document_id,
        "filename": document.filename,
        "status": "deleted",
    }


@router.post("/query")
def query_documents(
    request: QueryRequest,
    db: Session = Depends(get_db),
):
    start_time = perf_counter()

    document_exists = db.get(Document, request.document_id)

    if document_exists is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    def run_pipeline(limit: int):
        retrieval_limit = (
            max(limit * 2, 10)
            if is_list_question(request.query)
            else limit
        )

        candidates = hybrid_retrieve_chunks(
            db=db,
            query=request.query,
            document_id=request.document_id,
            limit=retrieval_limit,
        )

        results = rerank_chunks(
            query=request.query,
            chunks=candidates,
            limit=limit,
        )

        generation_context = prepare_generation_context(
            query=request.query,
            chunks=results,
            max_chunks=3,
        )

        answer = generate_answer(
            query=request.query,
            chunks=generation_context,
        )

        features = calculate_answer_features(
            answer=answer,
            results=results,
        )

        confidence = calculate_confidence(**features)

        return answer, confidence, results

    answer, confidence, results = run_pipeline(request.limit)

    decision = make_decision(confidence)
    retry = False

    if decision == "retry":
        retry = True

        answer, confidence, results = run_pipeline(
            request.limit * 2
        )

        decision = make_decision(confidence)

    human_review = needs_human_review(confidence)

    if human_review:
        decision = "human_review"

    latency_ms = round(
        (perf_counter() - start_time) * 1000,
        2,
    )

    audit_log = AuditLog(
        query=request.query,
        answer=answer,
        confidence=confidence,
        decision=decision,
        retry=retry,
        latency_ms=latency_ms,
    )

    db.add(audit_log)
    db.commit()

    return {
        "query": request.query,
        "answer": answer,
        "confidence": confidence,
        "decision": decision,
        "retry": retry,
        "human_review": human_review,
        "latency_ms": latency_ms,
        "results": [
            {
                key: value
                for key, value in result.items()
                if key not in {
                    "_rerank_evidence",
                    "_confidence_rerank_score",
                }
            }
            for result in results
        ],
    }