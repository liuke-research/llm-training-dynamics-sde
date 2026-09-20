"""Training diagnostics, not a theorem or a calibrated failure probability."""
__version__ = "0.2.0"
from .monitor import MonitorConfig, OnlineMonitor
__all__ = ["MonitorConfig", "OnlineMonitor"]
