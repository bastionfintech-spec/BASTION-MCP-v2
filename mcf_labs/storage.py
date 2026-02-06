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
        # Ensure base directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
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
            if self.storage_path.exists():
                type_dirs = [
                    d for d in self.storage_path.iterdir() 
                    if d.is_dir()
                ]
            else:
                type_dirs = []
        
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
    
    def save_report(self, report: Report) -> bool:
        """Save a report to storage"""
        try:
            # Create directory structure: data/reports/{type}/{year}/{month}/
            type_dir = self.storage_path / report.type.value
            year_dir = type_dir / report.generated_at.strftime("%Y")
            month_dir = year_dir / report.generated_at.strftime("%m")
            
            month_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = month_dir / f"{report.id}.json"
            
            with open(file_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
            
            logger.info(f"Saved report: {report.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save report {report.id}: {e}")
            return False
    
    def delete_report(self, report_id: str) -> bool:
        """Delete a report by ID"""
        # Parse report ID to find file
        parts = report_id.split("-")
        if len(parts) < 2:
            return False
        
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
            return False
        
        # Find the file - check all possible date patterns
        type_dir = self.storage_path / report_type
        if not type_dir.exists():
            return False
        
        # Search for the file
        for year_dir in type_dir.iterdir():
            if not year_dir.is_dir():
                continue
            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir():
                    continue
                file_path = month_dir / f"{report_id}.json"
                if file_path.exists():
                    try:
                        file_path.unlink()
                        logger.info(f"Deleted report: {report_id}")
                        return True
                    except Exception as e:
                        logger.error(f"Failed to delete report {report_id}: {e}")
                        return False
        
        return False
    
    def clear_bad_reports(self) -> int:
        """Delete reports with bad data indicators"""
        deleted = 0
        bad_indicators = ["UNKNOWN $0", "$0.00M", "NULL DATA", "NO POSITIONS"]
        
        reports = self.list_reports(limit=1000)
        
        for report in reports:
            is_bad = False
            
            # Check summary for bad indicators
            for indicator in bad_indicators:
                if indicator in report.summary:
                    is_bad = True
                    break
            
            # Check if whale report has no valid data
            if report.type == ReportType.WHALE_INTELLIGENCE:
                positions = report.sections.get("top_positions", [])
                valid_positions = [p for p in positions if p.get("size_usd", 0) > 0]
                if len(valid_positions) < 3:
                    is_bad = True
            
            if is_bad:
                if self.delete_report(report.id):
                    deleted += 1
                    logger.info(f"Cleaned up bad report: {report.id}")
        
        return deleted


# Global storage instance
_storage: Optional[ReportStorage] = None


def get_storage(storage_path: str = "data/reports") -> ReportStorage:
    """Get or create the global storage instance"""
    global _storage
    if _storage is None:
        _storage = ReportStorage(storage_path)
    return _storage


# Hybrid storage - uses Supabase if available, falls back to filesystem
class HybridStorage:
    """
    Hybrid storage that saves to both Supabase (persistent) and filesystem (cache).
    Reads from Supabase first if available.
    """
    
    def __init__(self, storage_path: str = "data/reports"):
        self.file_storage = ReportStorage(storage_path)
        
        # Try to init Supabase
        try:
            from .supabase_storage import get_supabase_storage
            self.supabase = get_supabase_storage()
        except Exception as e:
            logger.warning(f"Supabase not available: {e}")
            self.supabase = None
    
    @property
    def supabase_available(self) -> bool:
        return self.supabase is not None and self.supabase.is_available
    
    def save_report(self, report: Report) -> bool:
        """Save to both Supabase and filesystem"""
        # Always save to filesystem as cache
        file_saved = self.file_storage.save_report(report)
        
        # Also save to Supabase if available
        if self.supabase_available:
            self.supabase.save_report(report)
        
        return file_saved
    
    def get_report(self, report_id: str) -> Optional[Report]:
        """Get from Supabase first, fallback to filesystem"""
        if self.supabase_available:
            report = self.supabase.get_report(report_id)
            if report:
                return report
        
        return self.file_storage.get_report(report_id)
    
    def list_reports(
        self,
        report_type: Optional[ReportType] = None,
        limit: int = 50,
        offset: int = 0,
        since: Optional[datetime] = None,
        bias: Optional[str] = None
    ) -> List[Report]:
        """List from Supabase first, fallback to filesystem"""
        if self.supabase_available:
            reports = self.supabase.list_reports(
                report_type=report_type,
                bias=bias,
                limit=limit,
                offset=offset
            )
            if reports:
                return reports
        
        return self.file_storage.list_reports(
            report_type=report_type,
            limit=limit,
            offset=offset,
            since=since,
            bias=bias
        )
    
    def get_reports_for_research_terminal(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get formatted reports for UI"""
        if self.supabase_available:
            reports = self.supabase.get_reports_for_research_terminal(limit)
            if reports:
                return reports
        
        return self.file_storage.get_reports_for_research_terminal(limit)
    
    def get_latest_by_type(self) -> Dict[str, Optional[Report]]:
        """Get latest of each type"""
        if self.supabase_available:
            return self.supabase.get_latest_by_type()
        
        return self.file_storage.get_latest_by_type()
    
    def count_reports(self, report_type: Optional[ReportType] = None) -> int:
        """Count reports"""
        if self.supabase_available:
            return self.supabase.count_reports(report_type)
        
        return self.file_storage.count_reports(report_type)
    
    def delete_report(self, report_id: str) -> bool:
        """Delete from both storages"""
        file_deleted = self.file_storage.delete_report(report_id)
        
        if self.supabase_available:
            self.supabase.delete_report(report_id)
        
        return file_deleted
    
    def clear_bad_reports(self) -> int:
        """Clear bad reports from both storages"""
        file_deleted = self.file_storage.clear_bad_reports()
        
        if self.supabase_available:
            self.supabase.clear_bad_reports()
        
        return file_deleted
    
    def sync_to_supabase(self) -> int:
        """Sync all filesystem reports to Supabase"""
        if not self.supabase_available:
            logger.warning("Supabase not available for sync")
            return 0
        
        synced = 0
        reports = self.file_storage.list_reports(limit=1000)
        
        for report in reports:
            if self.supabase.save_report(report):
                synced += 1
        
        logger.info(f"[Hybrid] Synced {synced} reports to Supabase")
        return synced


# Global hybrid storage
_hybrid_storage: Optional[HybridStorage] = None


def get_hybrid_storage(storage_path: str = "data/reports") -> HybridStorage:
    """Get or create hybrid storage (Supabase + filesystem)"""
    global _hybrid_storage
    if _hybrid_storage is None:
        _hybrid_storage = HybridStorage(storage_path)
    return _hybrid_storage

