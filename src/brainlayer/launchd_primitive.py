"""Reusable launchd install and loaded-state verification primitives."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Callable

CommandRunner = Callable[[list[str]], Any]


def _default_command_runner(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(args=args, returncode=127, stdout="", stderr=str(exc))


def _command_returncode(result: Any) -> int:
    if isinstance(result, int):
        return result
    returncode = getattr(result, "returncode", None)
    if returncode is None:
        return 1
    try:
        return int(returncode)
    except (TypeError, ValueError):
        return 1


def _command_stdout(result: Any) -> str:
    return str(getattr(result, "stdout", "") or "")


def _command_stderr(result: Any) -> str:
    return str(getattr(result, "stderr", "") or "")


def launchd_target(label: str, *, uid: int | None = None) -> str:
    if not label:
        raise ValueError("launchd label must not be empty")
    return f"gui/{os.getuid() if uid is None else uid}/{label}"


class LaunchdVerificationError(RuntimeError):
    """Raised when launchd state cannot satisfy a required loaded post-condition."""

    def __init__(
        self,
        message: str,
        *,
        label: str,
        target: str,
        reason: str,
        plist_path: Path | None = None,
        command: list[str] | None = None,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        super().__init__(message)
        self.label = label
        self.target = target
        self.reason = reason
        self.plist_path = plist_path
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def issue_details(self) -> dict[str, Any]:
        details: dict[str, Any] = {
            "label": self.label,
            "target": self.target,
            "reason": self.reason,
        }
        if self.plist_path is not None:
            details["plist_path"] = str(self.plist_path)
        if self.command is not None:
            details["command"] = self.command
        if self.returncode is not None:
            details["returncode"] = self.returncode
        if self.stdout:
            details["stdout"] = self.stdout
        if self.stderr:
            details["stderr"] = self.stderr
        return details


class LaunchdLabelNotLoadedError(LaunchdVerificationError):
    """Raised when launchd reports that a required label is not loaded."""


class LaunchdCommandError(LaunchdVerificationError):
    """Raised when an install/bootstrap command fails before verification."""


def _loaded_state_from_result(result: Any) -> bool | None:
    returncode = _command_returncode(result)
    stdout = _command_stdout(result)
    stderr = _command_stderr(result)
    if returncode == 0:
        return True
    if returncode == 127:
        return None

    output = f"{stdout}\n{stderr}".lower()
    if "operation not permitted" in output or "permission" in output or "input/output error" in output:
        return None
    if "could not find service" in output or "service is not loaded" in output or returncode in {3, 36, 113}:
        return False
    return None


def is_launchd_label_loaded(
    label: str,
    *,
    command_runner: CommandRunner = _default_command_runner,
) -> bool | None:
    """Return True/False for known launchd loaded state, or None when indeterminate."""
    if not label:
        return True

    target = launchd_target(label)
    print_result = command_runner(["launchctl", "print", target])
    print_state = _loaded_state_from_result(print_result)
    if print_state is not None:
        return print_state
    return None


def verify_launchd_label_loaded(
    label: str,
    *,
    command_runner: CommandRunner = _default_command_runner,
    plist_path: Path | None = None,
    context: str = "",
) -> bool:
    """Require that launchd positively reports a label as loaded."""
    if not label:
        return True

    target = launchd_target(label)
    loaded = is_launchd_label_loaded(label, command_runner=command_runner)
    context_suffix = f" {context}" if context else ""
    if loaded is True:
        return True
    if loaded is False:
        raise LaunchdLabelNotLoadedError(
            f"launchd label {label} is not loaded{context_suffix}",
            label=label,
            target=target,
            reason="not_loaded",
            plist_path=plist_path,
        )
    raise LaunchdVerificationError(
        f"could not verify launchd label {label} is loaded{context_suffix}",
        label=label,
        target=target,
        reason="unknown_loaded_state",
        plist_path=plist_path,
    )


def _run_required_launchctl(
    args: list[str],
    *,
    label: str,
    plist_path: Path,
    command_runner: CommandRunner,
) -> None:
    result = command_runner(args)
    returncode = _command_returncode(result)
    if returncode == 0:
        return
    raise LaunchdCommandError(
        f"launchctl command failed for {label}: {' '.join(args)}",
        label=label,
        target=launchd_target(label),
        reason="launchctl_command_failed",
        plist_path=plist_path,
        command=args,
        returncode=returncode,
        stdout=_command_stdout(result),
        stderr=_command_stderr(result),
    )


def install_and_verify_launchagent(
    label: str,
    plist_path: Path,
    *,
    command_runner: CommandRunner = _default_command_runner,
    bootout_existing: bool = True,
) -> bool:
    """Bootstrap a LaunchAgent and require a positive loaded post-condition."""
    resolved_plist_path = plist_path.expanduser()
    target = launchd_target(label)
    domain = f"gui/{os.getuid()}"

    _run_required_launchctl(
        ["launchctl", "enable", target],
        label=label,
        plist_path=resolved_plist_path,
        command_runner=command_runner,
    )
    if bootout_existing:
        command_runner(["launchctl", "bootout", target])
    bootstrap_args = ["launchctl", "bootstrap", domain, str(resolved_plist_path)]
    bootstrap_result = command_runner(bootstrap_args)
    bootstrap_returncode = _command_returncode(bootstrap_result)
    if bootstrap_returncode != 0:
        try:
            return verify_launchd_label_loaded(
                label,
                command_runner=command_runner,
                plist_path=resolved_plist_path,
                context="after bootstrap",
            )
        except LaunchdVerificationError as exc:
            raise LaunchdCommandError(
                f"launchctl command failed for {label}: {' '.join(bootstrap_args)}",
                label=label,
                target=target,
                reason="launchctl_command_failed",
                plist_path=resolved_plist_path,
                command=bootstrap_args,
                returncode=bootstrap_returncode,
                stdout=_command_stdout(bootstrap_result),
                stderr=_command_stderr(bootstrap_result),
            ) from exc
    return verify_launchd_label_loaded(
        label,
        command_runner=command_runner,
        plist_path=resolved_plist_path,
        context="after bootstrap",
    )
