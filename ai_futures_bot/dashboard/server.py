"""Dependency-free dashboard HTTP server.

Serves the single-page UI from ``static/`` and exposes ``/api/state`` which
returns the latest JSON snapshot written by the backtester or live trader. The
front end polls ``/api/state`` and re-renders. No web framework required.
"""

from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from ..state import read_state

_STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
}


def _make_handler(state_path: str):
    class Handler(BaseHTTPRequestHandler):
        # Silence the default noisy logging.
        def log_message(self, *args) -> None:  # noqa: D401
            pass

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/state":
                self._send_state()
            elif path in ("/", "/index.html"):
                self._send_file("index.html")
            elif path.startswith("/static/"):
                self._send_file(path[len("/static/") :])
            else:
                self._send_file(path.lstrip("/"))

        def _send_state(self) -> None:
            state = read_state(state_path) or {
                "mode": "idle",
                "message": "No state yet. Run a backtest or live session first.",
            }
            body = json.dumps(state).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_file(self, rel: str) -> None:
            # Prevent path traversal.
            full = os.path.normpath(os.path.join(_STATIC_DIR, rel))
            if not full.startswith(_STATIC_DIR) or not os.path.isfile(full):
                self.send_error(404, "Not found")
                return
            ext = os.path.splitext(full)[1]
            ctype = _CONTENT_TYPES.get(ext, "application/octet-stream")
            with open(full, "rb") as fh:
                body = fh.read()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


class DashboardServer:
    def __init__(self, state_path: str, host: str = "127.0.0.1", port: int = 8000) -> None:
        self.state_path = state_path
        self.host = host
        self.port = port
        self._httpd: ThreadingHTTPServer | None = None

    def serve_forever(self) -> None:
        handler = _make_handler(self.state_path)
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        print(f"Dashboard: http://{self.host}:{self.port}  (state: {self.state_path})")
        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._httpd.server_close()

    def shutdown(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()


def serve(state_path: str, host: str = "127.0.0.1", port: int = 8000) -> None:
    DashboardServer(state_path, host, port).serve_forever()
