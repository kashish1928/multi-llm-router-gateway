"""loguru configuration with secret redaction."""

from __future__ import annotations

import re
import sys
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from loguru import Record

# Generic credential shapes, redacted even when not in the configured secret list.
_GENERIC_SECRET_RE = re.compile(
    r"\b(?:sk-or-v1-[A-Za-z0-9]{16,}|sk-[A-Za-z0-9_-]{20,}|pk-lf-[\w-]{8,}|sk-lf-[\w-]{8,})"
)
_BEARER_RE = re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}")


def redact(text: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            text = text.replace(secret, "[REDACTED]")
    text = _BEARER_RE.sub(r"\1[REDACTED]", text)
    return _GENERIC_SECRET_RE.sub("[REDACTED]", text)


def configure_logging(level: str, secrets: list[str], json_logs: bool = False) -> None:
    secret_list = sorted(set(secrets), key=len, reverse=True)

    def patcher(record: Record) -> None:
        record["message"] = redact(record["message"], secret_list)

    logger.remove()
    logger.configure(patcher=patcher)
    logger.add(sys.stderr, level=level.upper(), serialize=json_logs, backtrace=False, diagnose=False)
