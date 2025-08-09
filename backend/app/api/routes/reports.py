"""
Reports and analytics routes
"""

from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, and_, func
from pydantic import BaseModel

from app.core.database import get_db
from app.api.routes.auth import get_current_user
from app.models.user import User
from app.models.call import CallSession
from app.models.report import ScamReport, FeedbackReport

router = APIRouter()


class ScamReportCreate(BaseModel):
    session_id: str
    incident_type: str
    severity: str
    title: str
    description: str
    caller_number: Optional[str] = None
    caller_claimed_identity: Optional[str] = None


class FeedbackCreate(BaseModel):
    session_id: str
    feedback_type: str
    rating: Optional[int] = None
    is_actually_scam: Optional[bool] = None
    is_actually_synthetic: Optional[bool] = None
    user_comment: Optional[str] = None


@router.post("/scam-reports")
async def create_scam_report(
    report_data: ScamReportCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a scam incident report"""
    
    # Verify the session belongs to the user
    session_result = await db.execute(
        select(CallSession).where(
            CallSession.session_id == report_data.session_id,
            CallSession.user_id == current_user.id
        )
    )
    session = session_result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call session not found"
        )
    
    # Create the report
    report = ScamReport(
        session_id=report_data.session_id,
        user_id=current_user.id,
        report_type="user_reported",
        incident_type=report_data.incident_type,
        severity=report_data.severity,
        title=report_data.title,
        description=report_data.description,
        caller_number=report_data.caller_number,
        caller_claimed_identity=report_data.caller_claimed_identity,
        call_duration=session.call_duration,
        scam_score=session.scam_probability,
        spoofing_score=session.spoofing_probability,
        confidence_score=session.overall_risk_score,
        detected_language=session.language,
        incident_date=session.started_at,
        status="pending"
    )
    
    db.add(report)
    await db.commit()
    await db.refresh(report)
    
    return {
        "id": report.id,
        "message": "Scam report created successfully",
        "report_id": report.id,
        "status": report.status
    }


@router.get("/scam-reports")
async def get_scam_reports(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    status_filter: Optional[str] = Query(None),
    severity_filter: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's scam reports"""
    
    query = select(ScamReport).where(ScamReport.user_id == current_user.id)
    
    if status_filter:
        query = query.where(ScamReport.status == status_filter)
    
    if severity_filter:
        query = query.where(ScamReport.severity == severity_filter)
    
    query = query.order_by(desc(ScamReport.created_at)).offset(skip).limit(limit)
    
    result = await db.execute(query)
    reports = result.scalars().all()
    
    return [
        {
            "id": report.id,
            "session_id": report.session_id,
            "incident_type": report.incident_type,
            "severity": report.severity,
            "title": report.title,
            "description": report.description,
            "caller_number": report.caller_number,
            "caller_claimed_identity": report.caller_claimed_identity,
            "scam_score": report.scam_score,
            "spoofing_score": report.spoofing_score,
            "confidence_score": report.confidence_score,
            "detected_language": report.detected_language,
            "status": report.status,
            "is_confirmed_scam": report.is_confirmed_scam,
            "incident_date": report.incident_date,
            "created_at": report.created_at,
            "reviewed_at": report.reviewed_at
        }
        for report in reports
    ]


@router.get("/scam-reports/{report_id}")
async def get_scam_report(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get specific scam report details"""
    
    result = await db.execute(
        select(ScamReport).where(
            ScamReport.id == report_id,
            ScamReport.user_id == current_user.id
        )
    )
    report = result.scalar_one_or_none()
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Scam report not found"
        )
    
    return {
        "id": report.id,
        "session_id": report.session_id,
        "report_type": report.report_type,
        "incident_type": report.incident_type,
        "severity": report.severity,
        "title": report.title,
        "description": report.description,
        "key_indicators": report.key_indicators,
        "caller_number": report.caller_number,
        "caller_claimed_identity": report.caller_claimed_identity,
        "call_duration": report.call_duration,
        "scam_score": report.scam_score,
        "spoofing_score": report.spoofing_score,
        "confidence_score": report.confidence_score,
        "detected_language": report.detected_language,
        "transcript_excerpt": report.transcript_excerpt,
        "has_audio_evidence": report.has_audio_evidence,
        "evidence_metadata": report.evidence_metadata,
        "is_confirmed_scam": report.is_confirmed_scam,
        "false_positive": report.false_positive,
        "status": report.status,
        "reviewed_by": report.reviewed_by,
        "review_notes": report.review_notes,
        "incident_date": report.incident_date,
        "created_at": report.created_at,
        "reviewed_at": report.reviewed_at
    }


@router.post("/feedback")
async def submit_feedback(
    feedback_data: FeedbackCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Submit feedback on detection accuracy"""
    
    # Verify the session belongs to the user
    session_result = await db.execute(
        select(CallSession).where(
            CallSession.session_id == feedback_data.session_id,
            CallSession.user_id == current_user.id
        )
    )
    session = session_result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call session not found"
        )
    
    # Create feedback record
    feedback = FeedbackReport(
        session_id=feedback_data.session_id,
        user_id=current_user.id,
        feedback_type=feedback_data.feedback_type,
        rating=feedback_data.rating,
        original_scam_score=session.scam_probability,
        original_spoofing_score=session.spoofing_probability,
        original_risk_level=session.risk_level,
        is_actually_scam=feedback_data.is_actually_scam,
        is_actually_synthetic=feedback_data.is_actually_synthetic,
        user_comment=feedback_data.user_comment
    )
    
    db.add(feedback)
    await db.commit()
    await db.refresh(feedback)
    
    return {
        "id": feedback.id,
        "message": "Feedback submitted successfully",
        "feedback_id": feedback.id
    }


@router.get("/export")
async def export_user_data(
    format: str = Query("json", regex="^(json|csv)$"),
    days: int = Query(30, ge=1, le=365),
    include_transcripts: bool = Query(False),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Export user's call data"""
    
    from_date = datetime.utcnow() - timedelta(days=days)
    
    # Get call sessions
    sessions_result = await db.execute(
        select(CallSession).where(
            CallSession.user_id == current_user.id,
            CallSession.created_at >= from_date
        ).order_by(desc(CallSession.created_at))
    )
    sessions = sessions_result.scalars().all()
    
    # Get reports
    reports_result = await db.execute(
        select(ScamReport).where(
            ScamReport.user_id == current_user.id,
            ScamReport.created_at >= from_date
        ).order_by(desc(ScamReport.created_at))
    )
    reports = reports_result.scalars().all()
    
    # Prepare data
    export_data = {
        "export_metadata": {
            "user_id": current_user.id,
            "username": current_user.username,
            "export_date": datetime.utcnow().isoformat(),
            "period_days": days,
            "total_sessions": len(sessions),
            "total_reports": len(reports)
        },
        "call_sessions": [
            {
                "session_id": session.session_id,
                "caller_number": session.caller_number,
                "caller_name": session.caller_name,
                "call_duration": session.call_duration,
                "language": session.language,
                "status": session.status,
                "overall_risk_score": session.overall_risk_score,
                "scam_probability": session.scam_probability,
                "spoofing_probability": session.spoofing_probability,
                "risk_level": session.risk_level,
                "is_scam": session.is_scam,
                "is_synthetic": session.is_synthetic,
                "has_alerts": session.has_alerts,
                "started_at": session.started_at.isoformat(),
                "ended_at": session.ended_at.isoformat() if session.ended_at else None,
                "created_at": session.created_at.isoformat()
            }
            for session in sessions
        ],
        "scam_reports": [
            {
                "id": report.id,
                "session_id": report.session_id,
                "incident_type": report.incident_type,
                "severity": report.severity,
                "title": report.title,
                "description": report.description,
                "caller_number": report.caller_number,
                "scam_score": report.scam_score,
                "spoofing_score": report.spoofing_score,
                "status": report.status,
                "is_confirmed_scam": report.is_confirmed_scam,
                "created_at": report.created_at.isoformat()
            }
            for report in reports
        ]
    }
    
    if format == "json":
        from fastapi.responses import JSONResponse
        return JSONResponse(
            content=export_data,
            headers={
                "Content-Disposition": f"attachment; filename=scam_shield_data_{current_user.username}_{datetime.utcnow().strftime('%Y%m%d')}.json"
            }
        )
    
    elif format == "csv":
        import csv
        import io
        from fastapi.responses import StreamingResponse
        
        # Create CSV content
        output = io.StringIO()
        
        # Write sessions CSV
        if sessions:
            writer = csv.DictWriter(output, fieldnames=export_data["call_sessions"][0].keys())
            writer.writeheader()
            writer.writerows(export_data["call_sessions"])
        
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=scam_shield_sessions_{current_user.username}_{datetime.utcnow().strftime('%Y%m%d')}.csv"
            }
        )


@router.get("/statistics")
async def get_user_statistics(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get comprehensive user statistics"""
    
    from_date = datetime.utcnow() - timedelta(days=days)
    
    # Overall statistics
    stats_result = await db.execute(
        select(
            func.count(CallSession.id).label('total_calls'),
            func.sum(func.cast(CallSession.is_scam, 'INTEGER')).label('scam_calls'),
            func.sum(func.cast(CallSession.is_synthetic, 'INTEGER')).label('synthetic_calls'),
            func.avg(CallSession.overall_risk_score).label('avg_risk_score'),
            func.sum(CallSession.call_duration).label('total_duration')
        ).where(
            CallSession.user_id == current_user.id,
            CallSession.created_at >= from_date
        )
    )
    stats = stats_result.first()
    
    # Monthly trends
    monthly_result = await db.execute(
        select(
            func.date_trunc('month', CallSession.created_at).label('month'),
            func.count(CallSession.id).label('calls'),
            func.sum(func.cast(CallSession.is_scam, 'INTEGER')).label('scam_calls')
        ).where(
            CallSession.user_id == current_user.id,
            CallSession.created_at >= from_date
        ).group_by('month').order_by('month')
    )
    monthly_trends = [
        {
            "month": row.month.strftime('%Y-%m'),
            "total_calls": row.calls,
            "scam_calls": row.scam_calls or 0
        }
        for row in monthly_result.all()
    ]
    
    return {
        "period_days": days,
        "overall_statistics": {
            "total_calls": stats.total_calls or 0,
            "scam_calls": stats.scam_calls or 0,
            "synthetic_calls": stats.synthetic_calls or 0,
            "average_risk_score": float(stats.avg_risk_score or 0),
            "total_call_duration_minutes": int((stats.total_duration or 0) / 60),
            "scam_detection_rate": (
                (stats.scam_calls or 0) / (stats.total_calls or 1) * 100
                if stats.total_calls else 0
            )
        },
        "monthly_trends": monthly_trends,
        "generated_at": datetime.utcnow()
    }
