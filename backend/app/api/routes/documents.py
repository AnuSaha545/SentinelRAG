from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.document import Document, DocumentChunk
from app.services.document_processor import process_document
from app.services.retrieval import hybrid_retrieve_chunks, rerank_chunks
from app.services.generation import generate_answer
from app.services.confidence import calculate_confidence


router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


class QueryRequest(BaseModel):
    query: str
    document_id: str
    limit: int = 5


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


@router.post("/query")
def query_documents(
    request: QueryRequest,
    db: Session = Depends(get_db),
):
    candidates = hybrid_retrieve_chunks(
        db=db,
        query=request.query,
        document_id=request.document_id,
        limit=request.limit,
    )

    results = rerank_chunks(
        query=request.query,
        chunks=candidates,
        limit=request.limit,
    )

    confidence = calculate_confidence(results)

    answer = generate_answer(
        query=request.query,
        chunks=results,
    )

    return {
        "query": request.query,
        "answer": answer,
        "confidence": confidence,
        "results": results,
    }