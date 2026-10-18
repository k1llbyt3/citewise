
"""Simple process-local pacing for external API requests."""

import os
import threading
import time

_lock = threading.Lock()
_last_request_started = 0.0


def wait_for_api_slot() -> None:
    """Keep API request starts at least the configured interval apart."""
    global _last_request_started

    raw_interval = os.getenv("CITEWISE_MIN_API_INTERVAL_SECONDS", "1.0")

    try:
        interval = float(raw_interval)
    except ValueError as exc:
        raise ValueError(
            "CITEWISE_MIN_API_INTERVAL_SECONDS must be a number."
        ) from exc

    if not 0.0 <= interval <= 60.0:
        raise ValueError(
            "CITEWISE_MIN_API_INTERVAL_SECONDS must be between 0 and 60."
        )

    with _lock:
        now = time.monotonic()
        remaining = interval - (now - _last_request_started)

        if remaining > 0:
            time.sleep(remaining)

        _last_request_started = time.monotonic()
