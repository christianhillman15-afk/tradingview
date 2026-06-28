"""Triple-barrier labeling (López de Prado).

Naive "sign of the return H bars ahead" labels ignore the *path*: a trade that
would have been stopped out and then recovered gets a winning label it never
could have realised. Triple-barrier labeling fixes this by placing three
barriers from each bar and labeling by **which is touched first**:

  * upper barrier  = entry + ``pt_mult`` × ATR   (profit-take)
  * lower barrier  = entry − ``sl_mult`` × ATR   (stop-loss)
  * vertical barrier = ``max_horizon`` bars later (time stop)

Label = +1 if the upper barrier is touched first, −1 if the lower barrier is
first, 0 if the time barrier is hit first. It also returns ``t1`` — the bar index
where each label *resolves* — which purged cross-validation needs to remove
overlapping (leaky) training samples.

Pure stdlib.
"""

from __future__ import annotations

from typing import Sequence

from ..data import Bar, atr_inputs
from ..indicators import atr


def triple_barrier_labels(
    bars: Sequence[Bar],
    *,
    pt_mult: float = 2.0,
    sl_mult: float = 2.0,
    max_horizon: int = 20,
    atr_period: int = 14,
) -> tuple[list[int | None], list[int]]:
    """Return ``(labels, t1)`` aligned to ``bars``.

    ``labels[i]`` is +1 / −1 / 0, or ``None`` during ATR warm-up or when there is
    no room for the vertical barrier. ``t1[i]`` is the bar index where bar ``i``'s
    label resolves (== i when undefined).
    """
    highs, lows, closes = atr_inputs(bars)
    a = atr(highs, lows, closes, atr_period)
    n = len(bars)
    labels: list[int | None] = [None] * n
    t1: list[int] = list(range(n))
    for i in range(n):
        if a[i] is None or i + 1 >= n:
            continue
        entry = closes[i]
        upper = entry + pt_mult * a[i]
        lower = entry - sl_mult * a[i]
        end = min(i + max_horizon, n - 1)
        label = 0
        resolved = end
        for j in range(i + 1, end + 1):
            hit_up = highs[j] >= upper
            hit_dn = lows[j] <= lower
            if hit_up and hit_dn:
                # Both barriers inside the same bar; assume the adverse one first.
                label, resolved = -1, j
                break
            if hit_up:
                label, resolved = 1, j
                break
            if hit_dn:
                label, resolved = -1, j
                break
        labels[i] = label
        t1[i] = resolved
    return labels, t1


def binary_labels(labels: Sequence[int | None]) -> list[int | None]:
    """Collapse triple-barrier labels to up/down (drop the 0 / time-barrier
    class) for a binary directional classifier: +1 -> 1, -1 -> 0, 0 -> None."""
    out: list[int | None] = []
    for v in labels:
        if v is None or v == 0:
            out.append(None)
        else:
            out.append(1 if v > 0 else 0)
    return out
