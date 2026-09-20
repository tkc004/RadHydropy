# Copyright (C) 2026 Tsang Keung Chan
# SPDX-License-Identifier: AGPL-3.0
"""Structured logging helpers for runtime diagnostics.

RadHydropy is a library, so it does not configure handlers or write directly
to stdout. Applications can configure the ``radhydropy`` logger and consume
the event name and ``diagnostic_fields`` from each log record.
"""

import logging
from typing import Any

LOGGER = logging.getLogger("radhydropy")


def log_diagnostic(level: int, event: str, **fields: Any) -> None:
    """Emit a structured diagnostic record without configuring logging."""
    LOGGER.log(
        level,
        event,
        extra={
            "event": event,
            "diagnostic_fields": fields,
        },
    )
