"""
MCF Labs Report Storage
========================
Filesystem-based storage with optional Supabase sync.

Reports are stored as JSON files organized by type/year/month.
If Supabase credentials are configured, reports are also synced
to the mcf_reports table for cloud persistence.
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
    """Filesystem-based report storage"""

    # Map report ID prefixes to report types
    PREFIX_MAP = {
        "MS": "market_structure",
        "WI": "whale_intelligence",
        "OF": "options_flow",
        "CP": "cycle_position",
        "FA": "funding_arbitrage",
        "LC": "liquidation_cascade",
        "IR": "institutional_research",
    }

    def __init__(self, storage_path: str = "data/reports"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"[Storage] Initialized at {self.storage_path.absolute()}")

    def _type_from_id(self, report_id: str) -> Optional[str]:
        """Extract report type from ID prefix (e.g., MS-BTC-20260206-00 -> market_structure)"""
        prefix = report_id.split("-")[0] if "-" in report_id else ""
        return self.PREFIX_MAP.get(prefix)

    def get_report(self, report_id: str) -> Optional[Report]:
        """Get a specific report by ID"""
        report_type = self._type_from_id(report_id)
        if not report_type:
            return None

        type_dir = self.storage_path / report_type
        if not type_dir.exists():
            return None

        # Search through year/month dirs
        for year_dir in sorted(type_dir.iterdir(), reverse=True):
            if not year_dir.is_dir():
                continue
            for month_dir in sorted(year_dir.iterdir(), reverse=True):
                if not month_dir.is_dir():
                    continue
                file_path = month_dir / f"{report_id}.json"
                if file_path.exists():
                    try:
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        return Report.from_dict(data)
                    except Exception as e:
                        logger.error(f"[Storage] Failed to load {file_path}: {e}")
                        return None

        return None

    def list_reports(
        self,
        report_type: Optional[ReportType] = None,
        limit: int = 20,
        offset: int = 0,
        since: Optional[datetime] = None,
        bias: Optional[str] = None,
    ) -> List[Report]:
        """List reports with optional filters"""
        all_files = []

        # Determine which type directories to scan
        if report_type:
            type_dirs = [self.storage_path / report_type.value]
        else:
            type_dirs = [
                d for d in self.storage_path.iterdir()
                if d.is_dir() and d.name in [rt.value for rt in ReportType]
            ]

        for type_dir in type_dirs:
            if not type_dir.exists():
                continue
            for year_dir in type_dir.iterdir():
                if not year_dir.is_dir():
                    continue
                for month_dir in year_dir.iterdir():
                    if not month_dir.is_dir():
                        continue
                    for f in month_dir.glob("*.json"):
                        all_files.append(f)

        # Sort by modification time (newest first)
        all_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # Load and filter
        reports = []
        for file_path in all_files:
            if len(reports) >= offset + limit:
                break

            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                report = Report.from_dict(data)

                # Apply filters
                if since and report.generated_at < since:
                    continue
                if bias and report.bias.value != bias:
                    continue

                reports.append(report)
            except Exception:
                continue

        return reports[offset:offset + limit]

    def get_latest_by_type(self) -> Dict[str, Report]:
        """Get the latest report for each ReportType"""
        latest = {}
        for rt in ReportType:
            results = self.list_reports(report_type=rt, limit=1)
            latest[rt.value] = results[0] if results else None
        return latest

    def count_reports(
        self,
        report_type: Optional[ReportType] = None,
        since: Optional[datetime] = None,
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
        result = []
        for r in reports:
            result.append({
                "id": r.id,
                "type": r.type.value,
                "type_label": r.type.value.replace("_", " ").title(),
                "title": r.title,
                "summary": r.summary,
                "bias": r.bias.value,
                "confidence": r.confidence.value,
                "generated_at": r.generated_at.isoformat() + ("Z" if not r.generated_at.tzinfo else ""),
                "time_ago": self._time_ago(r.generated_at),
                "tags": r.tags,
            })
        return result

    def save_report(self, report: Report) -> bool:
        """Save a report to filesystem"""
        try:
            year = report.generated_at.strftime("%Y")
            month = report.generated_at.strftime("%m")
            dir_path = self.storage_path / report.type.value / year / month
            dir_path.mkdir(parents=True, exist_ok=True)

            file_path = dir_path / f"{report.id}.json"
            with open(file_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2)

            logger.info(f"[Storage] Saved report: {report.id}")
            return True
        except Exception as e:
            logger.error(f"[Storage] Failed to save {report.id}: {e}")
            return False

    def delete_report(self, report_id: str) -> bool:
        """Delete a report by ID"""
        report_type = self._type_from_id(report_id)
        if not report_type:
            return False

        type_dir = self.storage_path / report_type
        if not type_dir.exists():
            return False

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
                        logger.info(f"[Storage] Deleted report: {report_id}")
                        return True
                    except Exception as e:
                        logger.error(f"[Storage] Failed to delete {report_id}: {e}")
                        return False

        return False

    def clear_bad_reports(self) -> int:
        """Delete reports with bad data indicators"""
        bad_indicators = ["UNKNOWN $0", "$0.00M", "NULL DATA", "NO POSITIONS"]
        deleted = 0

        reports = self.list_reports(limit=1000)
        for report in reports:
            is_bad = False

            for indicator in bad_indicators:
                if indicator in report.summary:
                    is_bad = True
                    break

            # Check whale reports for valid positions
            if report.type == ReportType.WHALE_INTELLIGENCE:
                positions = report.sections.get("top_positions", [])
                valid = [p for p in positions if p.get("size_usd", 0) > 0]
                if len(valid) < 3:
                    is_bad = True

            if is_bad:
                if self.delete_report(report.id):
                    deleted += 1

        return deleted

    def _time_ago(self, dt: datetime) -> str:
        """Human-readable time-ago string"""
        try:
            now = datetime.utcnow()
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


class HybridStorage(ReportStorage):
    """
    Hybrid storage: filesystem + Supabase sync.

    Primary: Filesystem (always works, fast)
    Secondary: Supabase (cloud persistence, survives redeploys)

    Writes go to both. Reads prefer filesystem, fall back to Supabase.
    """

    def __init__(self, storage_path: str = "data/reports"):
        super().__init__(storage_path)
        self._supabase = None
        self._init_supabase()

    def _init_supabase(self):
        """Try to initialize Supabase connection"""
        try:
            from .supabase_storage import get_supabase_storage
            self._supabase = get_supabase_storage()
            if self._supabase.is_available:
                logger.info("[HybridStorage] Supabase connected — dual-write enabled")
            else:
                logger.info("[HybridStorage] Supabase not available — filesystem only")
                self._supabase = None
        except Exception as e:
            logger.warning(f"[HybridStorage] Supabase init failed: {e}")
            self._supabase = None

    @property
    def supabase_available(self) -> bool:
        return self._supabase is not None and self._supabase.is_available

    def save_report(self, report: Report) -> bool:
        """Save to filesystem + Supabase"""
        # Always save to filesystem
        fs_ok = super().save_report(report)

        # Also save to Supabase if available
        if self._supabase and self._supabase.is_available:
            try:
                self._supabase.save_report(report)
            except Exception as e:
                logger.warning(f"[HybridStorage] Supabase save failed for {report.id}: {e}")

        return fs_ok

    def get_report(self, report_id: str) -> Optional[Report]:
        """Try filesystem first, fall back to Supabase"""
        report = super().get_report(report_id)

        if report is None and self.supabase_available:
            try:
                report = self._supabase.get_report(report_id)
            except Exception:
                pass

        return report

    def delete_report(self, report_id: str) -> bool:
        """Delete from both stores"""
        fs_ok = super().delete_report(report_id)

        if self.supabase_available:
            try:
                self._supabase.delete_report(report_id)
            except Exception:
                pass

        return fs_ok

    def list_reports(
        self,
        report_type: Optional[ReportType] = None,
        limit: int = 20,
        offset: int = 0,
        since: Optional[datetime] = None,
        bias: Optional[str] = None,
    ) -> List[Report]:
        """List reports — filesystem first, fall back to Supabase if empty"""
        reports = super().list_reports(
            report_type=report_type, limit=limit, offset=offset,
            since=since, bias=bias
        )

        # If filesystem returned nothing, try Supabase
        if not reports and self.supabase_available:
            try:
                reports = self._supabase.list_reports(
                    report_type=report_type, limit=limit, offset=offset
                )
                logger.info(f"[HybridStorage] Supabase fallback returned {len(reports)} reports")
            except Exception as e:
                logger.warning(f"[HybridStorage] Supabase list_reports failed: {e}")

        return reports

    def get_latest_by_type(self) -> Dict[str, Report]:
        """Get latest report per type — filesystem first, fall back to Supabase"""
        latest = super().get_latest_by_type()

        # Check if any types are missing — fill from Supabase
        if self.supabase_available:
            missing = [k for k, v in latest.items() if v is None]
            if missing:
                try:
                    sb_latest = self._supabase.get_latest_by_type()
                    for key in missing:
                        if sb_latest.get(key):
                            latest[key] = sb_latest[key]
                            logger.info(f"[HybridStorage] Supabase filled latest for {key}")
                except Exception as e:
                    logger.warning(f"[HybridStorage] Supabase get_latest_by_type failed: {e}")

        return latest

    def get_reports_for_research_terminal(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get reports for Research Terminal UI — filesystem first, Supabase fallback"""
        results = super().get_reports_for_research_terminal(limit=limit)

        if not results and self.supabase_available:
            try:
                results = self._supabase.get_reports_for_research_terminal(limit=limit)
                logger.info(f"[HybridStorage] Supabase fallback returned {len(results)} research reports")
            except Exception as e:
                logger.warning(f"[HybridStorage] Supabase get_reports_for_research_terminal failed: {e}")

        return results

    def count_reports(
        self,
        report_type: Optional[ReportType] = None,
        since: Optional[datetime] = None,
    ) -> int:
        """Count reports — filesystem first, Supabase fallback"""
        count = super().count_reports(report_type=report_type, since=since)

        if count == 0 and self.supabase_available:
            try:
                count = self._supabase.count_reports(report_type=report_type)
            except Exception:
                pass

        return count


# Global instances
_storage: Optional[ReportStorage] = None
_hybrid_storage: Optional[HybridStorage] = None


def get_storage(storage_path: str = "data/reports") -> ReportStorage:
    """Get or create the global filesystem storage instance"""
    global _storage
    if _storage is None:
        _storage = ReportStorage(storage_path)
    return _storage


def get_hybrid_storage(storage_path: str = "data/reports") -> HybridStorage:
    """Get or create the global hybrid (filesystem + Supabase) storage instance"""
    global _hybrid_storage
    if _hybrid_storage is None:
        _hybrid_storage = HybridStorage(storage_path)
    return _hybrid_storage
