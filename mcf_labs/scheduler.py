"""
MCF Labs Report Scheduler
=========================
Automated scheduling for report generation
"""

import asyncio
import logging
import json
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Global scheduler instance
_scheduler = None
_running = False


class ReportScheduler:
    """Manages scheduled report generation"""
    
    def __init__(self, generator, storage_path: str = "data/reports"):
        self.generator = generator
        self.storage_path = storage_path
        self.tasks = []
        self._ensure_storage_dirs()
    
    def _ensure_storage_dirs(self):
        """Create storage directories if they don't exist"""
        report_types = [
            "market_structure",
            "whale_intelligence", 
            "options_flow",
            "cycle_position",
            "funding_arbitrage",
            "liquidation_cascade"
        ]
        for rt in report_types:
            path = os.path.join(self.storage_path, rt)
            os.makedirs(path, exist_ok=True)
    
    async def save_report(self, report):
        """Save report to filesystem"""
        date = report.generated_at
        year = date.strftime("%Y")
        month = date.strftime("%m")
        
        # Create directory structure
        dir_path = os.path.join(
            self.storage_path,
            report.type.value,
            year,
            month
        )
        os.makedirs(dir_path, exist_ok=True)
        
        # Save report
        file_path = os.path.join(dir_path, f"{report.id}.json")
        with open(file_path, "w") as f:
            f.write(report.to_json())
        
        logger.info(f"Saved report: {file_path}")
        return file_path
    
    async def run_market_structure(self):
        """Generate and save market structure report"""
        try:
            report = await self.generator.generate_market_structure("BTC")
            await self.save_report(report)
            logger.info(f"Generated market structure report: {report.id}")
            return report
        except Exception as e:
            logger.error(f"Failed to generate market structure report: {e}")
            return None
    
    async def run_whale_report(self):
        """Generate and save whale intelligence report"""
        try:
            report = await self.generator.generate_whale_report("BTC")
            await self.save_report(report)
            logger.info(f"Generated whale report: {report.id}")
            return report
        except Exception as e:
            logger.error(f"Failed to generate whale report: {e}")
            return None
    
    async def run_options_report(self):
        """Generate and save options flow report"""
        try:
            report = await self.generator.generate_options_report("BTC")
            await self.save_report(report)
            logger.info(f"Generated options report: {report.id}")
            return report
        except Exception as e:
            logger.error(f"Failed to generate options report: {e}")
            return None
    
    async def run_cycle_report(self):
        """Generate and save cycle position report"""
        try:
            report = await self.generator.generate_cycle_report("BTC")
            await self.save_report(report)
            logger.info(f"Generated cycle report: {report.id}")
            return report
        except Exception as e:
            logger.error(f"Failed to generate cycle report: {e}")
            return None
    
    async def check_liquidation_risk(self):
        """Check for high liquidation risk and generate alert if needed"""
        # TODO: Implement liquidation risk monitoring
        pass
    
    async def schedule_loop(self):
        """Main scheduling loop"""
        global _running
        _running = True
        
        logger.info("MCF Labs scheduler started")
        
        # Track last run times
        last_market_structure = None
        last_whale = None
        last_options = None
        last_cycle = None
        
        while _running:
            now = datetime.utcnow()
            hour = now.hour
            
            # Market Structure - every 4 hours (0, 4, 8, 12, 16, 20)
            if hour % 4 == 0 and last_market_structure != hour:
                await self.run_market_structure()
                last_market_structure = hour
            
            # Whale Intelligence - every 2 hours
            if hour % 2 == 0 and last_whale != hour:
                await self.run_whale_report()
                last_whale = hour
            
            # Options Flow - every 6 hours (0, 6, 12, 18)
            if hour % 6 == 0 and last_options != hour:
                await self.run_options_report()
                last_options = hour
            
            # Cycle Position - daily at 00:00
            if hour == 0 and last_cycle != now.date():
                await self.run_cycle_report()
                last_cycle = now.date()
            
            # Check liquidation risk every 5 minutes
            await self.check_liquidation_risk()
            
            # Sleep for 5 minutes
            await asyncio.sleep(300)
        
        logger.info("MCF Labs scheduler stopped")


async def start_scheduler(coinglass_client, helsinki_client=None, whale_alert_client=None):
    """Start the report scheduler"""
    global _scheduler
    
    from .generator import ReportGenerator
    
    generator = ReportGenerator(
        coinglass_client=coinglass_client,
        helsinki_client=helsinki_client,
        whale_alert_client=whale_alert_client
    )
    
    _scheduler = ReportScheduler(generator)
    
    # Start scheduler in background
    asyncio.create_task(_scheduler.schedule_loop())
    
    logger.info("MCF Labs report scheduler initialized")
    return _scheduler


async def stop_scheduler():
    """Stop the report scheduler"""
    global _running
    _running = False
    logger.info("Stopping MCF Labs scheduler...")


def get_scheduler() -> Optional[ReportScheduler]:
    """Get the current scheduler instance"""
    return _scheduler

