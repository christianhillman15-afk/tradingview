"""Web dashboard (pure-stdlib HTTP server + single-page tabbed UI)."""

from .server import DashboardServer, serve

__all__ = ["DashboardServer", "serve"]
