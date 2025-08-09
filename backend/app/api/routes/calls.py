"""
Call management routes
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc, func
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.api.routes.auth import get_current_user
from app.models.user import User
from app.models.call import CallSession, CallAnalysis, CallAlert, CallTranscript
from app.schemas.call import (
    CallSessionCreate, CallSessionUpdate, CallSessionResponse,
    CallTranscriptResponse, CallStatsResponse
)

router = APIRouter()


@router.post("/sessions", response_model=CallSessionResponse)
async def create_call_session(
    session_data: CallSessionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new call session"""
    call_session = CallSession(
        user_id=current_user.id,
        caller_number=session_data.caller_number,
        caller_name=session_data.caller_name,
        language=session_data.language,
        metadata=session_data.metadata,
        status="active"
    )
    
    db.add(call_session)
    await db.commit()
    await db.refresh(call_session)
    
    return CallSessionResponse.from_orm(call_session)


@router.get("/sessions", response_model=List[CallSessionResponse])
async def get_call_sessions(
    skip: int = 0,
    limit: int = 50,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get user's call sessions"""
    query = select(CallSession).where(CallSession.user_id == current_user.id)
    
    if status:
        query = query.where(CallSession.status == status)
    
    query = query.order_by(desc(CallSession.created_at)).offset(skip).limit(limit)
    
    result = await db.execute(query)
    sessions = result.scalars().all()
    
    return [CallSessionResponse.from_orm(session) for session in sessions]


@router.get("/sessions/{session_id}", response_model=CallSessionResponse)
async def get_call_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get specific call session"""
    result = await db.execute(
        select(CallSession).where(
            CallSession.session_id == session_id,
            CallSession.user_id == current_user.id
        )
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call session not found"
        )
    
    return CallSessionResponse.from_orm(session)


@router.put("/sessions/{session_id}", response_model=CallSessionResponse)
async def update_call_session(
    session_id: str,
    update_data: CallSessionUpdate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update call session"""
    result = await db.execute(
        select(CallSession).where(
            CallSession.session_id == session_id,
            CallSession.user_id == current_user.id
        )
    )
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call session not found"
        )
    
    # Update fields
    update_dict = update_data.dict(exclude_unset=True)
    for field, value in update_dict.items():
        setattr(session, field, value)
    
    # Calculate call duration if ending the call
    if update_data.status == "completed" and session.started_at:
        end_time = update_data.ended_at or datetime.utcnow()
        session.call_duration = int((end_time - session.started_at).total_seconds())
    
    await db.commit()
    await db.refresh(session)
    
    return CallSessionResponse.from_orm(session)


@router.get("/sessions/{session_id}/transcript", response_model=CallTranscriptResponse)
async def get_call_transcript(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get call transcript"""
    # First verify the session belongs to the user
    session_result = await db.execute(
        select(CallSession).where(
            CallSession.session_id == session_id,
            CallSession.user_id == current_user.id
        )
    )
    if not session_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call session not found"
        )
    
    # Get transcript
    result = await db.execute(
        select(CallTranscript).where(CallTranscript.session_id == session_id)
    )
    transcript = result.scalar_one_or_none()
    
    if not transcript:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcript not found"
        )
    
    return CallTranscriptResponse.from_orm(transcript)


@router.get("/sessions/{session_id}/analysis")
async def get_call_analysis(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed call analysis"""
    # Verify session ownership
    session_result = await db.execute(
        select(CallSession).where(
            CallSession.session_id == session_id,
            CallSession.user_id == current_user.id
        )
    )
    if not session_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call session not found"
        )
    
    # Get all analysis records for this session
    analysis_result = await db.execute(
        select(CallAnalysis).where(
            CallAnalysis.session_id == session_id
        ).order_by(CallAnalysis.timestamp)
    )
    analyses = analysis_result.scalars().all()
    
    # Get alerts for this session
    alerts_result = await db.execute(
        select(CallAlert).where(
            CallAlert.session_id == session_id
        ).order_by(CallAlert.created_at)
    )
    alerts = alerts_result.scalars().all()
    
    return {
        "session_id": session_id,
        "analyses": [
            {
                "id": analysis.id,
                "timestamp": analysis.timestamp,
                "transcription": analysis.transcription,
                "language": analysis.language,
                "scam_score": analysis.scam_score,
                "spoofing_score": analysis.spoofing_score,
                "risk_level": analysis.risk_level,
                "confidence": analysis.overall_confidence,
                "indicators": analysis.scam_indicators,
                "speaker": analysis.speaker
            }
            for analysis in analyses
        ],
        "alerts": [
            {
                "id": alert.id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.created_at,
                "acknowledged": alert.is_acknowledged
            }
            for alert in alerts
        ]
    }


@router.get("/sessions/{session_id}/alerts")
async def get_call_alerts(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get alerts for a specific call session"""
    # Verify session ownership
    session_result = await db.execute(
        select(CallSession).where(
            CallSession.session_id == session_id,
            CallSession.user_id == current_user.id
        )
    )
    if not session_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call session not found"
        )
    
    # Get alerts
    result = await db.execute(
        select(CallAlert).where(
            CallAlert.session_id == session_id
        ).order_by(desc(CallAlert.created_at))
    )
    alerts = result.scalars().all()
    
    return [
        {
            "id": alert.id,
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "risk_level": alert.risk_level,
            "title": alert.title,
            "message": alert.message,
            "language": alert.language,
            "is_acknowledged": alert.is_acknowledged,
            "is_dismissed": alert.is_dismissed,
            "created_at": alert.created_at,
            "metadata": alert.metadata
        }
        for alert in alerts
    ]


@router.post("/sessions/{session_id}/alerts/{alert_id}/acknowledge")
async def acknowledge_alert(
    session_id: str,
    alert_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Acknowledge an alert"""
    # Verify session ownership
    session_result = await db.execute(
        select(CallSession).where(
            CallSession.session_id == session_id,
            CallSession.user_id == current_user.id
        )
    )
    if not session_result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call session not found"
        )
    
    # Get and update alert
    result = await db.execute(
        select(CallAlert).where(
            CallAlert.id == alert_id,
            CallAlert.session_id == session_id
        )
    )
    alert = result.scalar_one_or_none()
    
    if not alert:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Alert not found"
        )
    
    alert.is_acknowledged = True
    alert.acknowledged_at = datetime.utcnow()
    
    await db.commit()
    
    return {"message": "Alert acknowledged"}


@router.get("/stats", response_model=CallStatsResponse)
async def get_call_stats(
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get call statistics for the user"""
    from_date = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    from_date = from_date - timedelta(days=days)
    
    # Total calls
    total_calls_result = await db.execute(
        select(func.count(CallSession.id)).where(
            CallSession.user_id == current_user.id,
            CallSession.created_at >= from_date
        )
    )
    total_calls = total_calls_result.scalar() or 0
    
    # Scam calls detected
    scam_calls_result = await db.execute(
        select(func.count(CallSession.id)).where(
            CallSession.user_id == current_user.id,
            CallSession.created_at >= from_date,
            CallSession.is_scam == True
        )
    )
    scam_calls = scam_calls_result.scalar() or 0
    
    # Synthetic voices detected
    synthetic_calls_result = await db.execute(
        select(func.count(CallSession.id)).where(
            CallSession.user_id == current_user.id,
            CallSession.created_at >= from_date,
            CallSession.is_synthetic == True
        )
    )
    synthetic_calls = synthetic_calls_result.scalar() or 0
    
    # Average risk score
    avg_risk_result = await db.execute(
        select(func.avg(CallSession.overall_risk_score)).where(
            CallSession.user_id == current_user.id,
            CallSession.created_at >= from_date
        )
    )
    avg_risk = avg_risk_result.scalar() or 0.0
    
    # Language distribution
    lang_result = await db.execute(
        select(CallSession.language, func.count(CallSession.id)).where(
            CallSession.user_id == current_user.id,
            CallSession.created_at >= from_date
        ).group_by(CallSession.language)
    )
    languages_detected = {lang: count for lang, count in lang_result.all()}
    
    # Risk level distribution
    risk_result = await db.execute(
        select(CallSession.risk_level, func.count(CallSession.id)).where(
            CallSession.user_id == current_user.id,
            CallSession.created_at >= from_date
        ).group_by(CallSession.risk_level)
    )
    risk_level_distribution = {level: count for level, count in risk_result.all()}
    
    return CallStatsResponse(
        total_calls=total_calls,
        scam_calls_detected=scam_calls,
        synthetic_voices_detected=synthetic_calls,
        average_risk_score=float(avg_risk),
        languages_detected=languages_detected,
        risk_level_distribution=risk_level_distribution
    )
