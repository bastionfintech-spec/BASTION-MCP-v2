"""
MCF Labs Report Scheduler
==========================
Automated report generation on schedule.

Generates market_structure, whale_intelligence, options_flow,
and cycle_position reports at configurable intervals.
"""

import asyncio
import logging
import json
import os
from datetime import datetime
from typing import Optional, List

logger = logging.getLogger(__name__)

# Global scheduler state
_scheduler: Optional["ReportScheduler"] = None
_running: bool = False

SUPPORTED_COINS = ("BTC", "ETH", "SOL", "BNB", "XRP", "AVAX", "LINK", "ARB", "OP")


class ReportScheduler:
    """Manages scheduled report generation"""

    def __init__(self, generator, storage_path: str = "data/reports"):
        self.generator = generator
        self.storage_path = storage_path
        self.tasks = []
        self._ensure_storage_dirs()

    def _ensure_storage_dirs(self):
        """Create storage directories for each report type"""
        from .models import ReportType
        for rt in ReportType:
            dir_path = os.path.join(self.storage_path, rt.value)
            os.makedirs(dir_path, exist_ok=True)

    def save_report(self, report) -> bool:
        """Save report to filesystem"""
        try:
            year = report.generated_at.strftime("%Y")
            month = report.generated_at.strftime("%m")
            dir_path = os.path.join(
                self.storage_path,
                report.type.value,
                year,
                month
            )
            os.makedirs(dir_path, exist_ok=True)

            file_path = os.path.join(dir_path, f"{report.id}.json")
            with open(file_path, "w") as f:
                f.write(report.to_json())

            logger.info(f"[Scheduler] Saved report: {report.id}")

            # Also sync to hybrid storage if available
            try:
                from .storage import get_hybrid_storage
                hybrid = get_hybrid_storage()
                if hybrid.supabase_available:
                    hybrid.save_report(report)
            except Exception:
                pass

            return True
        except Exception as e:
            logger.error(f"[Scheduler] Failed to save report {report.id}: {e}")
            return False

    async def run_market_structure(self, symbol: str = "BTC"):
        """Generate a market structure report"""
        try:
            logger.info(f"[Scheduler] Generating market structure report for {symbol}")
            report = await self.generator.generate_market_structure(symbol)
            if report:
                self.save_report(report)
                return report
        except Exception as e:
            logger.error(f"[Scheduler] Market structure generation failed for {symbol}: {e}")
        return None

    async def run_all_market_structure(self) -> List:
        """Generate market structure reports for all supported coins"""
        reports = []
        for symbol in SUPPORTED_COINS:
            try:
                report = await self.run_market_structure(symbol)
                if report:
                    reports.append(report)
                await asyncio.sleep(2)  # Rate limit
            except Exception as e:
                logger.error(f"[Scheduler] Error generating MS for {symbol}: {e}")
        return reports

    async def run_whale_report(self, symbol: str = "BTC"):
        """Generate a whale intelligence report"""
        try:
            logger.info(f"[Scheduler] Generating whale report for {symbol}")
            report = await self.generator.generate_whale_report(symbol)
            if report:
                self.save_report(report)
                return report
        except Exception as e:
            logger.error(f"[Scheduler] Whale report generation failed for {symbol}: {e}")
        return None

    async def run_all_whale_reports(self) -> List:
        """Generate whale reports for all supported coins"""
        reports = []
        for symbol in SUPPORTED_COINS:
            try:
                report = await self.run_whale_report(symbol)
                if report:
                    reports.append(report)
                await asyncio.sleep(2)
            except Exception as e:
                logger.error(f"[Scheduler] Error generating WI for {symbol}: {e}")
        return reports

    async def run_options_report(self, symbol: str = "BTC"):
        """Generate an options flow report"""
        try:
            logger.info(f"[Scheduler] Generating options report for {symbol}")
            report = await self.generator.generate_options_report(symbol)
            if report:
                self.save_report(report)
                return report
        except Exception as e:
            logger.error(f"[Scheduler] Options report generation failed for {symbol}: {e}")
        return None

    async def run_cycle_report(self, symbol: str = "BTC"):
        """Generate a cycle position report"""
        try:
            logger.info(f"[Scheduler] Generating cycle report for {symbol}")
            report = await self.generator.generate_cycle_report(symbol)
            if report:
                self.save_report(report)
                return report
        except Exception as e:
            logger.error(f"[Scheduler] Cycle report generation failed for {symbol}: {e}")
        return None

    async def check_liquidation_risk(self):
        """Check for liquidation risk conditions (stub)"""
        pass

    async def schedule_loop(self):
        """Main scheduling loop"""
        global _running

        last_market_structure = None
        last_whale = None
        last_options = None
        last_cycle = None

        logger.info("[Scheduler] Starting report generation loop")

        while _running:
            try:
                now = datetime.utcnow()
                hour = now.hour

                # Market structure: every 4 hours
                if last_market_structure is None or (now - last_market_structure).seconds >= 14400:
                    logger.info("[Scheduler] Running market structure reports...")
                    await self.run_all_market_structure()
                    last_market_structure = now

                # Whale intelligence: every 6 hours
                if last_whale is None or (now - last_whale).seconds >= 21600:
                    logger.info("[Scheduler] Running whale intelligence reports...")
                    await self.run_all_whale_reports()
                    last_whale = now

                # Options flow: every 8 hours
                if last_options is None or (now - last_options).seconds >= 28800:
                    logger.info("[Scheduler] Running options flow reports...")
                    for symbol in SUPPORTED_COINS[:3]:  # Top 3 only
                        await self.run_options_report(symbol)
                        await asyncio.sleep(2)
                    last_options = now

                # Cycle position: every 12 hours
                if last_cycle is None or (now - last_cycle).seconds >= 43200:
                    logger.info("[Scheduler] Running cycle position report...")
                    await self.run_cycle_report("BTC")
                    last_cycle = now

                # Sleep 5 minutes between checks
                await asyncio.sleep(300)

            except Exception as e:
                logger.error(f"[Scheduler] Loop error: {e}")
                await asyncio.sleep(60)

        logger.info("[Scheduler] Schedule loop stopped")


async def start_scheduler(
    coinglass_client,
    helsinki_client,
    whale_alert_client=None,
    use_iros: bool = True,
    model_url: str = None,
    model_api_key: str = None,
) -> ReportScheduler:
    """
    Start the report scheduler.

    If use_iros and model_url are provided, uses IROS-powered generation.
    Otherwise falls back to rule-based generation.
    """
    global _scheduler, _running

    # Create generator
    generator = None

    if use_iros and model_url:
        try:
            from .iros_generator import create_iros_generator
            generator = create_iros_generator(
                coinglass_client=coinglass_client,
                helsinki_client=helsinki_client,
                whale_alert_client=whale_alert_client,
                model_url=model_url,
                model_api_key=model_api_key,
            )
            logger.info("[Scheduler] Using IROS-powered report generator")
        except Exception as e:
            logger.warning(f"[Scheduler] IROS generator failed, falling back to rule-based: {e}")

    if generator is None:
        from .generator import ReportGenerator
        generator = ReportGenerator(
            coinglass_client=coinglass_client,
            helsinki_client=helsinki_client,
            whale_alert_client=whale_alert_client,
        )
        logger.info("[Scheduler] Using rule-based report generator")

    # Create scheduler
    _scheduler = ReportScheduler(generator)
    _running = True

    # Start schedule loop as background task
    asyncio.create_task(_scheduler.schedule_loop())
    logger.info("[Scheduler] Report scheduler started")

    return _scheduler


def stop_scheduler():
    """Stop the report scheduler"""
    global _running
    _running = False
    logger.info("[Scheduler] Stopping report scheduler")


def get_scheduler() -> Optional[ReportScheduler]:
    """Get the current scheduler instance"""
    return _scheduler
