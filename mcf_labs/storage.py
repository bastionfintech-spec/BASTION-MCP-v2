"""
MCF Labs Report Storage
=======================
Query and retrieve stored reports
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from pathlib import Path

from .models import Report, ReportType

logger = logging.getLogger(__name__)


class ReportStorage:
    """Query interface for stored MCF Labs reports"""
    
    def __init__(self, storage_path: str = "data/reports"):
        self.storage_path = Path(storage_path)
    
    def get_report(self, report_id: str) -> Optional[Report]:
        """Get a specific report by ID"""
        # Parse report ID to find file
        # Format: {TYPE}-{YYYYMMDD}-{HH} or {TYPE}-{YYYYMMDD}
        parts = report_id.split("-")
        if len(parts) < 2:
            return None
        
        type_prefix = parts[0]
        type_map = {
            "MS": "market_structure",
            "WI": "whale_intelligence",
            "OF": "options_flow",
            "CP": "cycle_position",
            "FA": "funding_arbitrage",
            "LC": "liquidation_cascade"
        }
        
        report_type = type_map.get(type_prefix)
        if not report_type:
            return None
        
        # Extract date
        date_str = parts[1]
        year = date_str[:4]
        month = date_str[4:6]
        
        file_path = self.storage_path / report_type / year / month / f"{report_id}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                return Report.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load report {report_id}: {e}")
            return None
    
    def list_reports(
        self,
        report_type: Optional[ReportType] = None,
        limit: int = 20,
        offset: int = 0,
        since: Optional[datetime] = None,
        bias: Optional[str] = None
    ) -> List[Report]:
        """List reports with optional filters"""
        reports = []
        
        # Determine which directories to scan
        if report_type:
            type_dirs = [self.storage_path / report_type.value]
        else:
            type_dirs = [
                d for d in self.storage_path.iterdir() 
                if d.is_dir()
            ]
        
        # Collect all report files
        report_files = []
        for type_dir in type_dirs:
            if not type_dir.exists():
                continue
            
            for year_dir in type_dir.iterdir():
                if not year_dir.is_dir():
                    continue
                
                for month_dir in year_dir.iterdir():
                    if not month_dir.is_dir():
                        continue
                    
                    for file in month_dir.glob("*.json"):
                        report_files.append(file)
        
        # Sort by modification time (newest first)
        report_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Load and filter reports
        for file in report_files:
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    report = Report.from_dict(data)
                    
                    # Apply filters
                    if since and report.generated_at < since:
                        continue
                    
                    if bias and report.bias.value != bias:
                        continue
                    
                    reports.append(report)
                    
                    # Check limit
                    if len(reports) >= offset + limit:
                        break
                        
            except Exception as e:
                logger.warning(f"Failed to load report {file}: {e}")
                continue
        
        # Apply offset and limit
        return reports[offset:offset + limit]
    
    def get_latest_by_type(self) -> Dict[str, Optional[Report]]:
        """Get the most recent report of each type"""
        latest = {}
        
        for report_type in ReportType:
            reports = self.list_reports(report_type=report_type, limit=1)
            latest[report_type.value] = reports[0] if reports else None
        
        return latest
    
    def count_reports(
        self,
        report_type: Optional[ReportType] = None,
        since: Optional[datetime] = None
    ) -> int:
        """Count reports matching filters"""
        return len(self.list_reports(
            report_type=report_type,
            limit=10000,
            since=since
        ))
    
    def get_reports_for_research_terminal(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get reports formatted for the Research Terminal UI"""
        reports = self.list_reports(limit=limit)
        
        return [
            {
                "id": r.id,
                "type": r.type.value,
                "type_label": r.type.value.replace("_", " ").title(),
                "title": r.title,
                "summary": r.summary,
                "bias": r.bias.value,
                "confidence": r.confidence.value,
                "generated_at": r.generated_at.isoformat(),
                "time_ago": self._time_ago(r.generated_at),
                "tags": r.tags
            }
            for r in reports
        ]
    
    def _time_ago(self, dt: datetime) -> str:
        """Format datetime as 'X min/hours/days ago'"""
        now = datetime.utcnow()
        diff = now - dt
        
        if diff.days > 0:
            return f"{diff.days}d ago"
        elif diff.seconds > 3600:
            return f"{diff.seconds // 3600}h ago"
        elif diff.seconds > 60:
            return f"{diff.seconds // 60}m ago"
        return "just now"


# Global storage instance
_storage: Optional[ReportStorage] = None


def get_storage(storage_path: str = "data/reports") -> ReportStorage:
    """Get or create the global storage instance"""
    global _storage
    if _storage is None:
        _storage = ReportStorage(storage_path)
    return _storage

