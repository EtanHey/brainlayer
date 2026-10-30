"""Explicit consolidated launcher and component CLI commands."""

import asyncio
import datetime as dt
import json
import sys
from pathlib import Path

import typer


def register(app, jobs_app):
    @app.command("maintenance-calendar")
    def maintenance_calendar(
        lib_dir: Path = typer.Option(Path(__file__).parent / "launchd", "--lib-dir"),
        scheduled_date: str | None = typer.Option(None, "--date"),
        state_dir: Path | None = typer.Option(None, "--state-dir"),
        logs_dir: Path = typer.Option(Path.home() / "Library/Logs/brainlayer", "--logs-dir"),
        recover_transcripts: bool = typer.Option(False, "--recover-transcripts"),
        recover_failed: bool = typer.Option(False, "--recover-failed"),
        status: bool = typer.Option(
            False, "--status", help="Check failed-step recovery admission without running steps"
        ),
    ):
        """Run BrainLayer Maintenance for one scheduled date."""
        from .maintenance_calendar import calendar_steps, run_calendar
        from .paths import supervisor_socket_path

        try:
            date = dt.date.fromisoformat(scheduled_date) if scheduled_date else dt.date.today()
        except ValueError as exc:
            raise typer.BadParameter("date must be ISO YYYY-MM-DD") from exc
        if recover_failed and recover_transcripts:
            raise typer.BadParameter("Choose --recover-failed or --recover-transcripts")
        if status and not recover_failed:
            raise typer.BadParameter("--status requires --recover-failed")
        if recover_failed and scheduled_date:
            raise typer.BadParameter("Failed-step recovery uses the latest scheduled night; omit --date")
        result = {}
        directory = state_dir or supervisor_socket_path().parent / "calendar"
        if directory.resolve() == supervisor_socket_path().parent.resolve():
            raise typer.BadParameter("calendar custody directory must differ from resident supervisor")
        try:
            plan = calendar_steps(sys.executable, str(Path(sys.executable).with_name("brainlayer")), lib_dir, date)
            code = asyncio.run(
                run_calendar(
                    plan,
                    directory,
                    logs_dir,
                    install_signals=True,
                    transcript_recovery=recover_transcripts,
                    **(
                        dict(failed_recovery=True, recovery_status=status, recovery_result=result)
                        if recover_failed
                        else {}
                    ),
                )
            )
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            if not recover_failed:
                raise
            from .maintenance_calendar import recovery_refusal_reason

            results = result.get("results") or {}
            if result.get("status") == "accepted" and results and all(v == "ok" for v in results.values()):
                # The retried steps finished and were journaled; only cleanup after them failed.
                result.update(
                    reason="Retried successfully; maintenance cleanup afterwards had a problem (see the log)."
                )
                code = 0
            else:
                status = "failed" if result.get("status") in {"accepted", "failed"} else "refused"
                result.update(status=status, reason=recovery_refusal_reason(exc), steps=[])
                code = 1
        if recover_failed:
            typer.echo(json.dumps(result, sort_keys=True))
        raise typer.Exit(code)

    @app.command("daemon")
    def daemon(manifest: Path = typer.Option(..., "--manifest"), socket: Path | None = typer.Option(None, "--socket")):
        """Run BrainLayer Daemon from a private command manifest."""
        from .supervisor_jobs import run_daemon

        run_daemon(manifest, socket)

    @jobs_app.command("component")
    def component(
        name: str,
        action: str,
        manifest: Path | None = typer.Option(None, "--manifest"),
        socket: Path | None = typer.Option(None, "--socket"),
    ):
        """Status/restart or configured env/log paths for one component."""
        from .component_jobs import component_action

        try:
            result = component_action(name, action, manifest=manifest, path=socket)
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            typer.echo(json.dumps({"error": str(exc)}))
            raise typer.Exit(1) from exc
        typer.echo(json.dumps(result, sort_keys=True))
