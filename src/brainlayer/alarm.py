"""Loud persistent alarms for never-silent degradation paths."""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, NoReturn

logger = logging.getLogger(__name__)

ALARM_DATASET = "brainlayer-alarms"


class BrainLayerAlarm(BaseException):
    """Alarm that bypasses broad ``except Exception`` handlers."""

    def __init__(
        self,
        code: str,
        message: str,
        context: dict[str, Any] | None = None,
        *,
        severity: str = "fatal",
        exit_code: int = 1,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.context = dict(context or {})
        self.severity = severity
        self.exit_code = exit_code

    @property
    def details(self) -> dict[str, Any]:
        return self.context

    def to_event(self) -> dict[str, Any]:
        return {
            "_type": "alarm",
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "context": self.context,
            "exit_code": self.exit_code,
        }

    def human_message(self) -> str:
        context = ""
        if self.context:
            context = f" context={json.dumps(self.context, sort_keys=True, default=str)}"
        return f"BRAINLAYER_ALARM {self.code}: {self.message}{context}"


def build_alarm(
    code: str,
    message: str,
    context: dict[str, Any] | None = None,
    *,
    severity: str = "fatal",
    exit_code: int = 1,
) -> BrainLayerAlarm:
    return BrainLayerAlarm(code, message, context, severity=severity, exit_code=exit_code)


def emit_alarm(alarm: BrainLayerAlarm) -> bool:
    """Persist the alarm to every existing non-blocking notification path."""
    human_message = alarm.human_message()
    logger.critical(human_message)
    try:
        print(human_message, file=sys.stderr, flush=True)
    except Exception as exc:
        logger.debug("Alarm stderr emit failed: %s", exc)
    try:
        from .telemetry import emit

        return emit(ALARM_DATASET, alarm.to_event())
    except Exception as exc:
        logger.debug("Alarm telemetry emit failed: %s", exc)
        return False


def raise_alarm(
    code: str,
    message: str,
    context: dict[str, Any] | None = None,
    *,
    severity: str = "fatal",
    exit_code: int = 1,
) -> NoReturn:
    alarm = build_alarm(code, message, context, severity=severity, exit_code=exit_code)
    emit_alarm(alarm)
    raise alarm
