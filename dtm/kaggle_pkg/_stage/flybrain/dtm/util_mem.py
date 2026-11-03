"""Wall-clock + peak RSS helpers."""
from __future__ import annotations

import resource
import time
from contextlib import contextmanager
from typing import Dict, Iterator


def rss_mb() -> float:
    # Linux: ru_maxrss is KB
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


@contextmanager
def track() -> Iterator[Dict]:
    t0 = time.perf_counter()
    rss0 = rss_mb()
    info: Dict = {}
    try:
        yield info
    finally:
        info["seconds"] = time.perf_counter() - t0
        info["peak_rss_mb"] = max(rss0, rss_mb())
