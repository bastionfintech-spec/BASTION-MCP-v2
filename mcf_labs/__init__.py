"""
MCF Labs Report Generation System
=================================
Automated institutional-grade crypto research reports.

Supports both rule-based and IROS-powered report generation.

Usage:
    # Start scheduler with IROS
    from mcf_labs import start_scheduler
    await start_scheduler(
        coinglass_client=coinglass,
        helsinki_client=helsinki,
        use_iros=True,
        model_url="https://your-vast-instance.trycloudflare.com"
    )
    
    # Generate single report
    from mcf_labs import IROSReportGenerator, create_iros_generator
    generator = create_iros_generator(coinglass, helsinki, model_url=...)
    report = await generator.generate_market_structure("BTC")
"""

from .scheduler import start_scheduler, stop_scheduler, get_scheduler, SUPPORTED_COINS
from .generator import ReportGenerator
from .iros_generator import IROSReportGenerator, create_iros_generator
from .models import Report, ReportType, Bias, Confidence, TradeScenario
from .storage import get_storage, ReportStorage

__all__ = [
    # Scheduler
    "start_scheduler",
    "stop_scheduler",
    "get_scheduler",
    "SUPPORTED_COINS",
    # Generators
    "ReportGenerator",
    "IROSReportGenerator", 
    "create_iros_generator",
    # Models
    "Report",
    "ReportType",
    "Bias",
    "Confidence",
    "TradeScenario",
    # Storage
    "get_storage",
    "ReportStorage",
]

