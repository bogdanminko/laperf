"""Reporting modules for La Perf benchmark suite."""

from .loaders import load_results
from .report import save_report

__all__ = ["save_report", "load_results"]
