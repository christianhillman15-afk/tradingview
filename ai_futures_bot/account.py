"""Persistent paper-trading account storage.

A paper account is a JSON document (cash, open position, trade history, indicator
buffer, feed RNG state, and metadata) that lets a live paper-trading session be
stopped and resumed without losing equity or open trades. This is what turns the
bot from a one-shot backtester into a standing $50k paper account you can leave
running and check on.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .state import read_state, write_state


def save_account(
    path: str,
    account: dict,
    *,
    symbol: str,
    strategy: str,
    created_at: str | None = None,
) -> None:
    """Atomically persist the account document with metadata."""
    doc = dict(account)
    doc["symbol"] = symbol
    doc["strategy"] = strategy
    doc["updated_at"] = datetime.now(timezone.utc).isoformat()
    doc["created_at"] = created_at or doc.get("created_at") or doc["updated_at"]
    write_state(path, doc)


def load_account(path: str) -> dict | None:
    """Load a persisted account, or ``None`` if it does not exist / is invalid."""
    return read_state(path)
