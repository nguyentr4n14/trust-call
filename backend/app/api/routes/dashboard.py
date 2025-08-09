"""
Dashboard and analytics routes
"""

from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc, and_

from app.core.database import get_db
from app.api.routes.auth import get_current_user
from app.models.user import User
from app.models.call import CallSession, CallAnalysis, CallAlert
from app.models.report import DetectionStats, AnalyticsEvent

router = APIRouter()


@router.get("/overview")
async def get_dashboard_overview(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get dashboard overview statistics"""
    from_date = datetime.utcnow() - timedelta(days=days)
    
    # Total calls
    total_calls_result = await db.execute(
        select(func.count(CallSession.id)).where(
            CallSession.user_id == current_user.id,
            CallSession.created_at >= from_date
        )
    )
    total_calls = total_calls_result.scalar() or 0
    
    # Active calls (currently ongoing)
    active_calls_result = await db.execute(
        select(func.count(CallSession.id)).where(
            CallSession.user_id == current_user.id,
            CallSession.status == "active"
        )
    )
    active_calls = active_calls_result.scalar() or 0
    
    # Scam calls detected
    scam_calls_result = await db.execute(
        select(func.count(CallSession.id)).where(
            CallSession.user_id == current_user.id,
            CallSession.created_at >= from_date,
            CallSession.is_scam == True
        )
    )
    scam_calls = scam_calls_result.scalar() or 0
    
    # High-risk calls
    high_risk_calls_result = await db.execute(
        select(func.count(CallSession.id)).where(
            CallSession.user_id == current_user.id,
            CallSession.created_at >= from_date,
            CallSession.risk_level.in_(["high", "critical"])
        )
    )
    high_risk_calls = high_risk_calls_result.scalar() or 0
    
    # Recent alerts
    recent_alerts_result = await db.execute(
        select(CallAlert).join(CallSession).where(
            CallSession.user_id == current_user.id,
            CallAlert.created_at >= from_date
        ).order_by(desc(CallAlert.created_at)).limit(10)
    )
    recent_alerts = recent_alerts_result.scalars().all()
    
    # Risk level distribution
    risk_distribution_result = await db.execute(
        select(CallSession.risk_level, func.count(CallSession.id)).where(
            CallSession.user_id == current_user.id,
            CallSession.created_at >= from_date
        ).group_by(CallSession.risk_level)
    )
    risk_distribution = dict(risk_distribution_result.all())
    
    # Language distribution
    language_distribution_result = await db.execute(
        select(CallSession.language, func.count(CallSession.id)).where(
            CallSession.user_id == current_user.id,
            CallSession.created_at >= from_date
        ).group_by(CallSession.language)
    )
    language_distribution = dict(language_distribution_result.all())
    
    # Calculate trends (comparing with previous period)
    previous_from_date = from_date - timedelta(days=days)
    
    previous_total_result = await db.execute(
        select(func.count(CallSession.id)).where(
            CallSession.user_id == current_user.id,
            CallSession.created_at >= previous_from_date,
            CallSession.created_at < from_date
        )
    )
    previous_total = previous_total_result.scalar() or 0
    
    previous_scam_result = await db.execute(
        select(func.count(CallSession.id)).where(
            CallSession.user_id == current_user.id,
            CallSession.created_at >= previous_from_date,
            CallSession.created_at < from_date,
            CallSession.is_scam == True
        )
    )
    previous_scam = previous_scam_result.scalar() or 0
    
    # Calculate percentage changes
    def calc_percentage_change(current, previous):
        if previous == 0:
            return 0 if current == 0 else 100
        return ((current - previous) / previous) * 100
    
    return {
        "period_days": days,
        "overview": {
            "total_calls": total_calls,
            "active_calls": active_calls,
            "scam_calls_detected": scam_calls,
            "high_risk_calls": high_risk_calls,
            "scam_detection_rate": (scam_calls / total_calls * 100) if total_calls > 0 else 0
        },
        "trends": {
            "calls_change_percent": calc_percentage_change(total_calls, previous_total),
            "scam_change_percent": calc_percentage_change(scam_calls, previous_scam)
        },
        "distributions": {
            "risk_levels": risk_distribution,
            "languages": language_distribution
        },
        "recent_alerts": [
            {
                "id": alert.id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "created_at": alert.created_at,
                "acknowledged": alert.is_acknowledged
            }
            for alert in recent_alerts
        ],
        "generated_at": datetime.utcnow()
    }


@router.get("/activity")
async def get_activity_timeline(
    days: int = Query(7, ge=1, le=30),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get activity timeline for the dashboard"""
    from_date = datetime.utcnow() - timedelta(days=days)
    
    # Daily call counts
    daily_calls_result = await db.execute(
        select(
            func.date(CallSession.created_at).label('date'),
            func.count(CallSession.id).label('total_calls'),
            func.sum(func.cast(CallSession.is_scam, db.bind.dialect.name == 'postgresql' and 'INTEGER' or 'INTEGER')).label('scam_calls')
        ).where(
            CallSession.user_id == current_user.id,
            CallSession.created_at >= from_date
        ).group_by(func.date(CallSession.created_at)).order_by('date')
    )
    
    daily_data = []
    for row in daily_calls_result.all():
        daily_data.append({
            "date": row.date.isoformat(),
            "total_calls": row.total_calls,
            "scam_calls": row.scam_calls or 0,
            "safe_calls": row.total_calls - (row.scam_calls or 0)
        })
    
    # Recent call sessions
    recent_sessions_result = await db.execute(
        select(CallSession).where(
            CallSession.user_id == current_user.id,
            CallSession.created_at >= from_date
        ).order_by(desc(CallSession.created_at)).limit(20)
    )
    recent_sessions = recent_sessions_result.scalars().all()
    
    return {
        "period_days": days,
        "daily_activity": daily_data,
        "recent_sessions": [
            {
                "session_id": session.session_id,
                "caller_number": session.caller_number,
                "caller_name": session.caller_name,
                "duration": session.call_duration,
                "risk_level": session.risk_level,
                "scam_score": session.scam_probability,
                "spoofing_score": session.spoofing_probability,
                "language": session.language,
                "status": session.status,
                "started_at": session.started_at,
                "has_alerts": session.has_alerts
            }
            for session in recent_sessions
        ],
        "generated_at": datetime.utcnow()
    }


@router.get("/alerts")
async def get_alerts_summary(
    days: int = Query(7, ge=1, le=30),
    severity: Optional[str] = Query(None),
    acknowledged: Optional[bool] = Query(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get alerts summary for the dashboard"""
    from_date = datetime.utcnow() - timedelta(days=days)
    
    # Build query
    query = select(CallAlert).join(CallSession).where(
        CallSession.user_id == current_user.id,
        CallAlert.created_at >= from_date
    )
    
    if severity:
        query = query.where(CallAlert.severity == severity)
    
    if acknowledged is not None:
        query = query.where(CallAlert.is_acknowledged == acknowledged)
    
    query = query.order_by(desc(CallAlert.created_at))
    
    result = await db.execute(query)
    alerts = result.scalars().all()
    
    # Group alerts by type and severity
    alert_summary = {}
    severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    type_counts = {}
    
    for alert in alerts:
        # Count by severity
        if alert.severity in severity_counts:
            severity_counts[alert.severity] += 1
        
        # Count by type
        alert_type = alert.alert_type
        if alert_type not in type_counts:
            type_counts[alert_type] = 0
        type_counts[alert_type] += 1
    
    return {
        "period_days": days,
        "summary": {
            "total_alerts": len(alerts),
            "unacknowledged_alerts": sum(1 for alert in alerts if not alert.is_acknowledged),
            "severity_distribution": severity_counts,
            "type_distribution": type_counts
        },
        "alerts": [
            {
                "id": alert.id,
                "session_id": alert.session_id,
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
            for alert in alerts[:50]  # Limit to recent 50 alerts
        ],
        "generated_at": datetime.utcnow()
    }


@router.get("/analytics")
async def get_analytics_data(
    metric: str = Query(..., regex="^(detection_accuracy|response_time|language_usage|risk_trends)$"),
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get specific analytics data"""
    from_date = datetime.utcnow() - timedelta(days=days)
    
    if metric == "detection_accuracy":
        # Calculate detection accuracy metrics
        # This would typically involve comparing with user feedback
        return {
            "metric": "detection_accuracy",
            "period_days": days,
            "data": {
                "overall_accuracy": 0.85,  # Placeholder
                "scam_detection_accuracy": 0.87,
                "false_positive_rate": 0.12,
                "false_negative_rate": 0.08
            },
            "generated_at": datetime.utcnow()
        }
    
    elif metric == "response_time":
        # Average response time for alerts
        return {
            "metric": "response_time",
            "period_days": days,
            "data": {
                "average_detection_time_ms": 1200,  # Placeholder
                "average_alert_time_ms": 800,
                "processing_time_trend": []  # Would contain daily averages
            },
            "generated_at": datetime.utcnow()
        }
    
    elif metric == "language_usage":
        # Language usage statistics
        language_result = await db.execute(
            select(CallSession.language, func.count(CallSession.id)).where(
                CallSession.user_id == current_user.id,
                CallSession.created_at >= from_date
            ).group_by(CallSession.language)
        )
        
        return {
            "metric": "language_usage",
            "period_days": days,
            "data": dict(language_result.all()),
            "generated_at": datetime.utcnow()
        }
    
    elif metric == "risk_trends":
        # Risk level trends over time
        weekly_risk_result = await db.execute(
            select(
                func.date_trunc('week', CallSession.created_at).label('week'),
                CallSession.risk_level,
                func.count(CallSession.id).label('count')
            ).where(
                CallSession.user_id == current_user.id,
                CallSession.created_at >= from_date
            ).group_by('week', CallSession.risk_level).order_by('week')
        )
        
        risk_trends = {}
        for row in weekly_risk_result.all():
            week_str = row.week.isoformat()
            if week_str not in risk_trends:
                risk_trends[week_str] = {}
            risk_trends[week_str][row.risk_level] = row.count
        
        return {
            "metric": "risk_trends",
            "period_days": days,
            "data": risk_trends,
            "generated_at": datetime.utcnow()
        }
