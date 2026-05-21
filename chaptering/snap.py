"""Snap chapter boundaries to the nearest TransNet shot change within a
tolerance window. Keeps the order of inputs and never produces a snap
that crosses a neighboring boundary."""

from __future__ import annotations

from bisect import bisect_left


def snap_boundaries_to_shots(
    boundary_times_ms: list[int],
    shot_change_times_ms: list[int],
    tolerance_ms: int,
) -> list[int]:
    """For each boundary, return the nearest shot-change time if it falls
    within `tolerance_ms`, else the original boundary. Order preserved."""
    if not shot_change_times_ms or tolerance_ms <= 0:
        return list(boundary_times_ms)

    shots = sorted(shot_change_times_ms)
    snapped: list[int] = []
    for t in boundary_times_ms:
        idx = bisect_left(shots, t)
        candidates = []
        if idx < len(shots):
            candidates.append(shots[idx])
        if idx > 0:
            candidates.append(shots[idx - 1])
        if not candidates:
            snapped.append(t)
            continue
        best = min(candidates, key=lambda s: abs(s - t))
        snapped.append(best if abs(best - t) <= tolerance_ms else t)
    return snapped
