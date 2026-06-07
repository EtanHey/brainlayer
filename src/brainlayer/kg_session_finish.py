"""One-command ORQI session finish pipeline.

The command harvests a voice-review session, gates each clean decision through
the KG entity judge, applies only judge-passing decisions through the reversible
cleanup applier, and writes judge failures back as an ORQI flag batch.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from brainlayer.kg_judge import judge_clusters_with_backend
from brainlayer.kg_session_harvest import harvest_session
from brainlayer.kg_session_harvest import main as harvest_main
from brainlayer.paths import get_db_path

SUMMARY_KEYS = (
    "harvested",
    "questions",
    "clean_decisions",
    "judge_pass",
    "judge_fail",
    "applied",
    "queued_for_review",
    "run_id",
    "reversible",
)

JudgeFunc = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


class ApplyFunc(Protocol):
    def __call__(self, path: Path, run_id: str, *, execute: bool) -> dict[str, Any]: ...


def finish_session(
    batch_path: str | Path,
    decisions_path: str | Path,
    *,
    run_id: str | None = None,
    no_judge: bool = False,
    dry_run: bool = False,
    judge_func: JudgeFunc | None = None,
    apply_func: ApplyFunc | None = None,
) -> dict[str, Any]:
    """Run the ORQI session-finish pipeline and return the summary contract."""
    batch = Path(batch_path)
    decisions = Path(decisions_path)
    resolved_run_id = run_id or _default_run_id(batch, decisions)

    if dry_run:
        with tempfile.TemporaryDirectory(prefix="brainlayer-session-finish-") as tmp:
            tmp_root = Path(tmp)
            harvest = harvest_session(
                batch,
                decisions,
                answers_path=tmp_root / f"{decisions.stem}.answers.json",
                decisions_path=tmp_root / f"{decisions.stem}.clean.json",
            )
            clean_path = tmp_root / f"{decisions.stem}.clean.json"
            clean = _load_json(clean_path)
            pass_items, fail_items = _judge_clean_decisions(
                clean,
                batch,
                no_judge=no_judge,
                judge_func=judge_func,
            )
    else:
        answers_path = decisions.with_name(f"{decisions.stem}.answers.json")
        clean_path = decisions.with_name(f"{decisions.stem}.clean.json")
        review_queue_path = decisions.with_name(f"{decisions.stem}.review-queue.json")
        passes_path = decisions.with_name(f"{decisions.stem}.passes.json")
        with contextlib.redirect_stdout(io.StringIO()):
            harvest_exit = harvest_main(
                [
                    "--batch",
                    str(batch),
                    "--session-decisions",
                    str(decisions),
                    "--answers",
                    str(answers_path),
                    "--decisions",
                    str(clean_path),
                ]
            )
        if harvest_exit != 0:
            raise RuntimeError(f"kg_session_harvest failed with exit code {harvest_exit}")
        harvest = {"answers_count": _count_answers(answers_path)}
        clean = _load_json(clean_path)
        pass_items, fail_items = _judge_clean_decisions(
            clean,
            batch,
            no_judge=no_judge,
            judge_func=judge_func,
        )
        if fail_items:
            _write_review_queue(fail_items, batch, review_queue_path)
        else:
            _unlink_if_exists(review_queue_path)
        if not pass_items:
            _unlink_if_exists(passes_path)

    applied = 0
    if not dry_run and pass_items:
        with _writer_pidfile_guard(get_db_path()):
            passes_doc = _filtered_decisions_doc(clean, pass_items)
            passes_path = decisions.with_name(f"{decisions.stem}.passes.json")
            passes_path.write_text(json.dumps(passes_doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (apply_func or _apply_passes)(passes_path, resolved_run_id, execute=True)
        applied = len(pass_items)

    summary = {
        "harvested": int(clean.get("counts", {}).get("merge_clusters", 0)),
        "questions": int(harvest["answers_count"]),
        "clean_decisions": len(_decision_items(clean)),
        "judge_pass": len(pass_items),
        "judge_fail": len(fail_items),
        "applied": applied,
        "queued_for_review": 0 if dry_run else len(fail_items),
        "run_id": resolved_run_id,
        "reversible": True,
    }
    return {key: summary[key] for key in SUMMARY_KEYS}


def _judge_clean_decisions(
    clean: dict[str, Any],
    batch_path: Path,
    *,
    no_judge: bool,
    judge_func: JudgeFunc | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items = _decision_items(clean)
    if no_judge:
        return items, []

    clusters = _clusters_by_key(batch_path)
    passes = []
    failures = []
    judge = judge_func or _default_judge
    for item in items:
        key = _item_key(item)
        cluster = clusters.get(key)
        if cluster is None:
            failures.append(item)
            continue
        verdict = judge(item, cluster)
        if _verdict_passes_item(item, verdict):
            passes.append(item)
        else:
            failures.append(item)
    return passes, failures


def _default_judge(item: dict[str, Any], cluster: dict[str, Any]) -> dict[str, Any]:
    del item
    return judge_clusters_with_backend([cluster], backend="groq")[0]


def _verdict_passes_item(item: dict[str, Any], verdict: dict[str, Any]) -> bool:
    if verdict.get("confidence") == "low" or verdict.get("evidence_degraded") is True:
        return False
    expected = "merge" if "canonical" in item else "keep"
    if verdict.get("merge_disposition") != expected:
        return False
    if expected == "merge":
        canonical = item.get("canonical")
        if not isinstance(canonical, dict):
            return False
        return verdict.get("canonical_suggestion") == canonical.get("name")
    return True


def _decision_items(clean: dict[str, Any]) -> list[dict[str, Any]]:
    return [*clean.get("merge", []), *clean.get("keep", [])]


def _filtered_decisions_doc(clean: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    wanted = {_item_key(item) for item in items}
    merge = [item for item in clean.get("merge", []) if _item_key(item) in wanted]
    keep = [item for item in clean.get("keep", []) if _item_key(item) in wanted]
    return {
        **clean,
        "counts": _counts(merge, keep),
        "per_category": _per_category(merge, keep),
        "merge": merge,
        "keep": keep,
    }


def _counts(merge: list[dict[str, Any]], keep: list[dict[str, Any]]) -> dict[str, int]:
    exported = merge + keep
    return {
        "merge_clusters": len(merge),
        "rows_merged_away": sum(len(item.get("members", [])) for item in merge),
        "keep": len(keep),
        "explicit": sum(1 for item in exported if item.get("source") not in {"rule", "voice-rule"}),
        "by_rule": sum(1 for item in exported if item.get("source") in {"rule", "voice-rule"}),
    }


def _per_category(merge: list[dict[str, Any]], keep: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    per_category: dict[str, dict[str, Any]] = {}
    for item in merge + keep:
        category = str(item.get("category") or "")
        row = per_category.setdefault(
            category,
            {"total": 0, "explicit": 0, "by_rule": 0, "undecided": 0, "rule": None},
        )
        row["total"] += 1
        if item.get("source") in {"rule", "voice-rule"}:
            row["by_rule"] += 1
        else:
            row["explicit"] += 1
    return per_category


def _write_review_queue(items: list[dict[str, Any]], batch_path: Path, out_path: Path) -> None:
    clusters = _clusters_by_key(batch_path)
    queue: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        cluster = clusters.get(_item_key(item))
        if cluster is None:
            cluster = {"stem": item.get("stem"), "members": [], "category": item.get("category")}
        category = str(cluster.get("category") or item.get("category") or "review")
        row = {key: value for key, value in cluster.items() if key != "category"}
        row["item_kind"] = _item_kind(row.get("members", []))
        queue.setdefault(category, []).append(row)
    out_path.write_text(json.dumps(queue, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _clusters_by_key(batch_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    raw = _load_json(batch_path)
    if not isinstance(raw, dict):
        raise ValueError("batch must be a JSON object keyed by category")
    clusters = {}
    for category, rows in raw.items():
        if not isinstance(rows, list):
            raise ValueError(f"batch category {category!r} must be a list")
        for row in rows:
            if not isinstance(row, dict):
                continue
            stem = row.get("stem")
            if isinstance(stem, str):
                clusters[(str(category), stem)] = {**row, "category": str(category)}
    return clusters


def _item_key(item: dict[str, Any]) -> tuple[str, str]:
    return str(item.get("category") or ""), str(item.get("stem") or "")


def _item_kind(members: Any) -> str:
    if isinstance(members, list) and any(
        isinstance(member, dict) and member.get("type") == "question" for member in members
    ):
        return "question"
    return "cluster"


def _apply_passes(path: Path, run_id: str, *, execute: bool) -> dict[str, Any]:
    script = Path(__file__).resolve().parents[2] / "scripts" / "kg_cleanup_apply.py"
    command = [sys.executable, str(script), "--decisions", str(path), "--run-id", run_id]
    if execute:
        command.append("--execute")
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "kg_cleanup_apply failed")
    return {"stdout": result.stdout}


def _default_run_id(batch_path: Path, decisions_path: Path) -> str:
    today = datetime.now(timezone.utc).date().isoformat()
    digest = hashlib.sha256()
    for path in (batch_path, decisions_path):
        digest.update(path.resolve().as_posix().encode("utf-8"))
        if path.exists():
            digest.update(path.read_bytes())
    return f"orqi-session-{today}-{digest.hexdigest()[:8]}"


def _writer_pidfile_path(db_path: str | Path) -> Path:
    pidfile_dir = Path(os.environ.get("BRAINLAYER_WRITER_PIDFILE_DIR", "/tmp")).expanduser()
    if not pidfile_dir.is_absolute():
        pidfile_dir = Path("/tmp") / pidfile_dir
    resolved_path = Path(db_path).expanduser().resolve()
    path_hash = hashlib.sha256(str(resolved_path).encode("utf-8")).hexdigest()[:16]
    return pidfile_dir.resolve() / f"brainlayer-writer-{path_hash}-{resolved_path.name}.pid"


@contextmanager
def _writer_pidfile_guard(db_path: str | Path):
    pidfile = _writer_pidfile_path(db_path)
    pidfile.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    fd = None
    for attempt in range(4):
        try:
            fd = os.open(pidfile, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            fcntl.flock(fd, fcntl.LOCK_EX)
            os.write(fd, f"{pid}\ndb_path={Path(db_path).expanduser().resolve()}\n".encode("utf-8"))
            break
        except FileExistsError as exc:
            owner_pid = _read_pidfile_owner(pidfile)
            if owner_pid is not None and _pid_is_alive(owner_pid):
                raise RuntimeError(f"another writer is using {db_path} (pid {owner_pid})") from exc
            try:
                pidfile.unlink()
            except FileNotFoundError:
                pass
            time.sleep(0.01 * (attempt + 1))
    else:
        raise RuntimeError(f"could not acquire writer pidfile for {db_path}")

    try:
        yield
    finally:
        if fd is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        if _read_pidfile_owner(pidfile) == pid:
            try:
                pidfile.unlink()
            except FileNotFoundError:
                pass


def _read_pidfile_owner(pidfile: Path) -> int | None:
    try:
        first = pidfile.read_text(encoding="utf-8").splitlines()[0].strip()
        return int(first)
    except (OSError, IndexError, ValueError):
        return None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _count_answers(answers_path: Path) -> int:
    json_path = answers_path if answers_path.suffix == ".json" else answers_path.with_suffix(".json")
    answers = _load_json(json_path)
    if not isinstance(answers, list):
        raise ValueError(f"answers output must be a JSON list: {json_path}")
    return len(answers)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Finish an ORQI KG review session.")
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.no_judge:
        print("kg_session_finish: --no-judge set; all clean decisions are treated as judge-passing", file=sys.stderr)

    with contextlib.redirect_stdout(io.StringIO()):
        summary = finish_session(
            args.batch,
            args.decisions,
            run_id=args.run_id,
            no_judge=args.no_judge,
            dry_run=args.dry_run,
        )
    print(json.dumps(summary, sort_keys=False))
    return 0


__all__ = ["finish_session", "main"]
