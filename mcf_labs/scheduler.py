"""
MCF Labs Report Scheduler
==========================
Automated report generation on schedule.

Generates institutional_research, market_structure, whale_intelligence,
options_flow, and cycle_position reports at configurable intervals.

Institutional reports are the primary output (Goldman-grade analysis).
Alert-style reports (market_structure, whale, etc.) run more frequently
as lightweight monitoring signals.
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

# Institutional reports focus on the top liquid coins
INSTITUTIONAL_COINS = ("BTC", "ETH", "SOL")


class ReportScheduler:
    """Manages scheduled report generation for all report types"""

    def __init__(self, generator, storage_path: str = "data/reports", institutional_generator=None):
        self.generator = generator
        self.institutional_generator = institutional_generator
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
        """Save report to filesystem and sync to Supabase"""
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

    # =========================================================================
    # INSTITUTIONAL REPORT GENERATION
    # =========================================================================

    async def run_institutional_report(self, symbol: str = "BTC"):
        """Generate an institutional research report"""
        if not self.institutional_generator:
            logger.warning("[Scheduler] No institutional generator configured")
            return None

        try:
            logger.info(f"[Scheduler] Generating institutional report for {symbol}")
            report = await self.institutional_generator.generate_institutional_report(symbol)
            if report:
                self.save_report(report)
                logger.info(f"[Scheduler] Institutional report generated: {report.id} (conviction: {report.sections.get('conviction', '?')}%)")
                return report
        except Exception as e:
            logger.error(f"[Scheduler] Institutional report failed for {symbol}: {e}")
        return None

    async def run_institutional_batch(self, symbols: tuple = INSTITUTIONAL_COINS) -> List:
        """Generate institutional reports for a batch of symbols"""
        reports = []
        for symbol in symbols:
            try:
                report = await self.run_institutional_report(symbol)
                if report:
                    reports.append(report)
                await asyncio.sleep(3)  # Rate limit between reports
            except Exception as e:
                logger.error(f"[Scheduler] Institutional batch error for {symbol}: {e}")
        logger.info(f"[Scheduler] Institutional batch complete: {len(reports)}/{len(symbols)} reports")
        return reports

    # =========================================================================
    # ALERT-STYLE REPORT GENERATION
    # =========================================================================

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

    # =========================================================================
    # MAIN SCHEDULING LOOP
    # =========================================================================

    async def schedule_loop(self):
        """
        Main scheduling loop.

        Schedule:
        - Institutional Research: every 6 hours (BTC, ETH, SOL)
        - Market Structure alerts: every 4 hours (all coins)
        - Whale Intelligence alerts: every 6 hours (all coins)
        - Options Flow alerts: every 8 hours (top 3)
        - Cycle Position: every 12 hours (BTC only)
        """
        global _running

        last_institutional = None
        last_market_structure = None
        last_whale = None
        last_options = None
        last_cycle = None

        logger.info("[Scheduler] Starting report generation loop")
        logger.info(f"[Scheduler] Institutional generator: {'ACTIVE' if self.institutional_generator else 'NOT CONFIGURED'}")

        while _running:
            try:
                now = datetime.utcnow()

                # Institutional Research: every 6 hours (flagship reports)
                if self.institutional_generator and (
                    last_institutional is None or (now - last_institutional).total_seconds() >= 21600
                ):
                    logger.info("[Scheduler] === GENERATING INSTITUTIONAL REPORTS ===")
                    await self.run_institutional_batch(INSTITUTIONAL_COINS)
                    last_institutional = now

                # Market structure alerts: every 4 hours
                if last_market_structure is None or (now - last_market_structure).total_seconds() >= 14400:
                    logger.info("[Scheduler] Running market structure alerts...")
                    await self.run_all_market_structure()
                    last_market_structure = now

                # Whale intelligence alerts: every 6 hours
                if last_whale is None or (now - last_whale).total_seconds() >= 21600:
                    logger.info("[Scheduler] Running whale intelligence alerts...")
                    await self.run_all_whale_reports()
                    last_whale = now

                # Options flow alerts: every 8 hours (top 3 only)
                if last_options is None or (now - last_options).total_seconds() >= 28800:
                    logger.info("[Scheduler] Running options flow alerts...")
                    for symbol in SUPPORTED_COINS[:3]:
                        await self.run_options_report(symbol)
                        await asyncio.sleep(2)
                    last_options = now

                # Cycle position: every 12 hours (BTC only)
                if last_cycle is None or (now - last_cycle).total_seconds() >= 43200:
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

    Creates both the alert generator (IROS or rule-based) and the
    institutional generator, then starts the scheduling loop.
    """
    global _scheduler, _running

    # Create alert generator
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
            logger.info("[Scheduler] Using IROS-powered alert generator")
        except Exception as e:
            logger.warning(f"[Scheduler] IROS generator failed, falling back to rule-based: {e}")

    if generator is None:
        from .generator import ReportGenerator
        generator = ReportGenerator(
            coinglass_client=coinglass_client,
            helsinki_client=helsinki_client,
            whale_alert_client=whale_alert_client,
        )
        logger.info("[Scheduler] Using rule-based alert generator")

    # Create institutional generator
    institutional_gen = None
    try:
        from .institutional_generator import create_institutional_generator
        institutional_gen = create_institutional_generator(
            coinglass_client=coinglass_client,
            helsinki_client=helsinki_client,
            whale_alert_client=whale_alert_client,
            model_url=model_url,
            model_api_key=model_api_key,
        )
        logger.info("[Scheduler] Institutional report generator initialized")
    except Exception as e:
        logger.warning(f"[Scheduler] Institutional generator init failed: {e}")

    # Create scheduler with both generators
    _scheduler = ReportScheduler(
        generator=generator,
        institutional_generator=institutional_gen,
    )
    _running = True

    # Start schedule loop as background task
    asyncio.create_task(_scheduler.schedule_loop())
    logger.info("[Scheduler] Report scheduler started (institutional + alerts)")

    return _scheduler


def stop_scheduler():
    """Stop the report scheduler"""
    global _running
    _running = False
    logger.info("[Scheduler] Stopping report scheduler")


def get_scheduler() -> Optional[ReportScheduler]:
    """Get the current scheduler instance"""
    return _scheduler
