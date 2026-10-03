from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.audit import AuditLog


router = APIRouter()


@router.get("/summary")
def audit_summary(
    db: Session = Depends(get_db),
):
    total_queries = db.scalar(
        select(func.count(AuditLog.id))
    ) or 0

    average_confidence = db.scalar(
        select(func.avg(AuditLog.confidence))
    )

    average_latency = db.scalar(
        select(func.avg(AuditLog.latency_ms))
    )

    retry_count = db.scalar(
        select(func.count(AuditLog.id))
        .where(AuditLog.retry.is_(True))
    ) or 0

    human_review_count = db.scalar(
        select(func.count(AuditLog.id))
        .where(AuditLog.decision == "human_review")
    ) or 0

    accepted_count = db.scalar(
        select(func.count(AuditLog.id))
        .where(AuditLog.decision == "accept")
    ) or 0

    return {
        "total_queries": total_queries,
        "average_confidence": round(
            float(average_confidence or 0.0),
            4,
        ),
        "average_latency_ms": round(
            float(average_latency or 0.0),
            2,
        ),
        "retry_count": retry_count,
        "human_review_count": human_review_count,
        "accepted_count": accepted_count,
    }


@router.get("/logs")
def audit_logs(
    db: Session = Depends(get_db),
):
    logs = db.scalars(
        select(AuditLog)
        .order_by(AuditLog.created_at.desc())
        .limit(100)
    ).all()

    return [
        {
            "id": log.id,
            "query": log.query,
            "answer": log.answer,
            "confidence": log.confidence,
            "decision": log.decision,
            "retry": log.retry,
            "latency_ms": log.latency_ms,
            "created_at": log.created_at,
        }
        for log in logs
    ]