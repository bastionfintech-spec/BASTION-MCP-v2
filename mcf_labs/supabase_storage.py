"""
MCF Labs Supabase Storage
=========================
Persistent report storage using Supabase.
Falls back to filesystem if Supabase not configured.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .models import Report, ReportType, Bias, Confidence

logger = logging.getLogger(__name__)

# Try to import supabase
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger.warning("Supabase not installed - using filesystem storage")


class SupabaseStorage:
    """Supabase-backed report storage for persistence across deploys"""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.table_name = "mcf_reports"
        self._init_client()
    
    def _init_client(self):
        """Initialize Supabase client"""
        if not SUPABASE_AVAILABLE:
            logger.warning("Supabase package not available")
            return
        
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            logger.warning("SUPABASE_URL or SUPABASE_KEY not set")
            return
        
        try:
            self.client = create_client(url, key)
            logger.info("[Supabase] Connected successfully")
        except Exception as e:
            logger.error(f"[Supabase] Connection failed: {e}")
    
    @property
    def is_available(self) -> bool:
        return self.client is not None
    
    def save_report(self, report: Report) -> bool:
        """Save report to Supabase"""
        if not self.is_available:
            return False
        
        try:
            data = {
                "id": report.id,
                "type": report.type.value,
                "title": report.title,
                "summary": report.summary,
                "bias": report.bias.value,
                "confidence": report.confidence.value,
                "symbol": report.symbol,
                "generated_at": report.generated_at.isoformat(),
                "tags": report.tags,
                "sections": report.sections,
                "scenarios": [s.__dict__ if hasattr(s, '__dict__') else s for s in report.scenarios] if report.scenarios else [],
                "full_content": json.dumps(report.to_dict())
            }
            
            # Upsert (insert or update)
            result = self.client.table(self.table_name).upsert(data).execute()
            logger.info(f"[Supabase] Saved report: {report.id}")
            return True
            
        except Exception as e:
            logger.error(f"[Supabase] Failed to save report {report.id}: {e}")
            return False
    
    def get_report(self, report_id: str) -> Optional[Report]:
        """Get a specific report by ID"""
        if not self.is_available:
            return None
        
        try:
            result = self.client.table(self.table_name)\
                .select("full_content")\
                .eq("id", report_id)\
                .single()\
                .execute()
            
            if result.data:
                data = json.loads(result.data["full_content"])
                return Report.from_dict(data)
            return None
            
        except Exception as e:
            logger.error(f"[Supabase] Failed to get report {report_id}: {e}")
            return None
    
    def list_reports(
        self,
        report_type: Optional[ReportType] = None,
        symbol: Optional[str] = None,
        bias: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Report]:
        """List reports with filters"""
        if not self.is_available:
            return []
        
        try:
            query = self.client.table(self.table_name)\
                .select("full_content")\
                .order("generated_at", desc=True)
            
            if report_type:
                query = query.eq("type", report_type.value)
            
            if symbol:
                query = query.eq("symbol", symbol)
            
            if bias:
                query = query.eq("bias", bias)
            
            query = query.range(offset, offset + limit - 1)
            
            result = query.execute()
            
            reports = []
            for row in result.data or []:
                try:
                    data = json.loads(row["full_content"])
                    reports.append(Report.from_dict(data))
                except:
                    continue
            
            return reports
            
        except Exception as e:
            logger.error(f"[Supabase] Failed to list reports: {e}")
            return []
    
    def get_reports_for_research_terminal(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get reports formatted for Research Terminal UI"""
        if not self.is_available:
            return []
        
        try:
            result = self.client.table(self.table_name)\
                .select("id, type, title, summary, bias, confidence, symbol, generated_at, tags")\
                .order("generated_at", desc=True)\
                .limit(limit)\
                .execute()
            
            reports = []
            for row in result.data or []:
                reports.append({
                    "id": row["id"],
                    "type": row["type"],
                    "type_label": row["type"].replace("_", " ").title(),
                    "title": row["title"],
                    "summary": row["summary"],
                    "bias": row["bias"],
                    "confidence": row["confidence"],
                    "symbol": row.get("symbol", "BTC"),
                    "generated_at": row["generated_at"],
                    "time_ago": self._time_ago(row["generated_at"]),
                    "tags": row.get("tags", [])
                })
            
            return reports
            
        except Exception as e:
            logger.error(f"[Supabase] Failed to get research terminal reports: {e}")
            return []
    
    def get_latest_by_type(self) -> Dict[str, Optional[Dict]]:
        """Get most recent report of each type"""
        if not self.is_available:
            return {}
        
        latest = {}
        for report_type in ReportType:
            try:
                result = self.client.table(self.table_name)\
                    .select("id, type, title, summary, bias, confidence, generated_at")\
                    .eq("type", report_type.value)\
                    .order("generated_at", desc=True)\
                    .limit(1)\
                    .execute()
                
                if result.data:
                    latest[report_type.value] = result.data[0]
                else:
                    latest[report_type.value] = None
                    
            except Exception as e:
                logger.error(f"[Supabase] Failed to get latest {report_type.value}: {e}")
                latest[report_type.value] = None
        
        return latest
    
    def count_reports(self, report_type: Optional[ReportType] = None) -> int:
        """Count total reports"""
        if not self.is_available:
            return 0
        
        try:
            query = self.client.table(self.table_name).select("id", count="exact")
            
            if report_type:
                query = query.eq("type", report_type.value)
            
            result = query.execute()
            return result.count or 0
            
        except Exception as e:
            logger.error(f"[Supabase] Failed to count reports: {e}")
            return 0
    
    def delete_report(self, report_id: str) -> bool:
        """Delete a report"""
        if not self.is_available:
            return False
        
        try:
            self.client.table(self.table_name).delete().eq("id", report_id).execute()
            logger.info(f"[Supabase] Deleted report: {report_id}")
            return True
        except Exception as e:
            logger.error(f"[Supabase] Failed to delete {report_id}: {e}")
            return False
    
    def clear_bad_reports(self) -> int:
        """Delete reports with bad data"""
        if not self.is_available:
            return 0
        
        deleted = 0
        bad_indicators = ["UNKNOWN $0", "$0.00M", "NULL DATA", "NO POSITIONS"]
        
        try:
            # Get all reports
            result = self.client.table(self.table_name)\
                .select("id, summary, type, full_content")\
                .execute()
            
            for row in result.data or []:
                is_bad = False
                
                for indicator in bad_indicators:
                    if indicator in row.get("summary", ""):
                        is_bad = True
                        break
                
                # Check whale reports for valid positions
                if row["type"] == "whale_intelligence":
                    try:
                        content = json.loads(row["full_content"])
                        positions = content.get("sections", {}).get("top_positions", [])
                        valid = [p for p in positions if p.get("size_usd", 0) > 0]
                        if len(valid) < 3:
                            is_bad = True
                    except:
                        pass
                
                if is_bad:
                    if self.delete_report(row["id"]):
                        deleted += 1
            
            return deleted
            
        except Exception as e:
            logger.error(f"[Supabase] Failed to clear bad reports: {e}")
            return 0
    
    def _time_ago(self, dt_str: str) -> str:
        """Format datetime string as 'X min/hours/days ago'"""
        try:
            dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            now = datetime.utcnow()
            
            # Make dt timezone-naive for comparison
            if dt.tzinfo:
                dt = dt.replace(tzinfo=None)
            
            diff = now - dt
            
            if diff.days > 0:
                return f"{diff.days}d ago"
            elif diff.seconds > 3600:
                return f"{diff.seconds // 3600}h ago"
            elif diff.seconds > 60:
                return f"{diff.seconds // 60}m ago"
            return "just now"
        except:
            return "recently"


# Global instance
_supabase_storage: Optional[SupabaseStorage] = None


def get_supabase_storage() -> SupabaseStorage:
    """Get or create the Supabase storage instance"""
    global _supabase_storage
    if _supabase_storage is None:
        _supabase_storage = SupabaseStorage()
    return _supabase_storage

