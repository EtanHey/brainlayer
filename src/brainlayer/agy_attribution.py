"""Which agent an Antigravity (agy) conversation ran as, decided before any of it is indexed.

agy, the current Gemini agent, records `agy --agent <name>` per conversation in its own
`~/.gemini/antigravity-cli/conversation_summaries.db` (`agent_name`). That row is the only
attribution read; task text never is. The gate fails closed so a memory reader is never indexed
as an interactive session:

* a row exists: index under that agent ("" = no agent); a memory reader is classed brain-worker
  downstream and never indexed;
* the DB was read and the row is absent: hold until `AGY_ATTRIBUTION_WAIT_S` after the FIRST time
  the conversation was seen unresolved (persisted, so restarts and appends do not extend it), then
  index with no agent;
* the DB could not be read: hold, retry on the next flush, and alarm `agy_attribution_unreadable`.

One race remains: a conversation indexed after its deadline whose row later names a memory reader.
Its later turns are not indexed, and the conversation is recorded under `late_memory_readers` in the
state file; chunks already written are not deleted.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any

from .alarm import BrainLayerAlarm, raise_alarm
from .ingest_denylist import MEMORY_READER_ATTRIBUTIONS

logger = logging.getLogger(__name__)

AGY_ATTRIBUTION_WAIT_S = 600.0
# State entries older than this are dropped. fallback_indexed only saves re-deciding a conversation;
# a re-decision is idempotent (content-hash chunk ids), it only holds new turns for one more deadline.
STATE_MAX_AGE_S = 30 * 24 * 3600.0
STATE_FILE_NAME = "agy-attribution.json"
_cache: dict[str, tuple[tuple[int, ...], dict[str, str]]] = {}


def _now() -> float:
    return time.time()


def agy_conversation(source_file: str | Path) -> tuple[Path, str] | None:
    """(summaries DB, conversation id) for a file under `.gemini/antigravity-cli/brain/<conversation>/`."""
    parts = Path(os.path.abspath(os.path.expanduser(str(source_file)))).parts
    for index, part in enumerate(parts[:-3]):
        if part == ".gemini" and parts[index + 1 : index + 3] == ("antigravity-cli", "brain"):
            return Path(*parts[: index + 2], "conversation_summaries.db"), parts[index + 3]
    return None


def read_conversation_agent(source_file: str | Path) -> tuple[bool, str | None]:
    """(read ok, agent name or None when the row is absent). A failed read is never cached."""
    located = agy_conversation(source_file)
    if located is None:
        return True, None
    store, conversation = located
    signature: list[int] = []
    for candidate in (store, store.with_name(store.name + "-wal")):
        try:
            stat = candidate.stat()
            signature += [stat.st_mtime_ns, stat.st_size]
        except OSError:
            signature += [0, 0]
    cached = _cache.get(str(store))
    if cached is not None and cached[0] == tuple(signature):
        return True, cached[1].get(conversation)
    try:
        conn = sqlite3.connect(f"file:{store}?mode=ro", uri=True, timeout=1.0)
        try:
            agents = {
                str(row_id): str(agent or "")
                for row_id, agent in conn.execute("SELECT conversation_id, agent_name FROM conversation_summaries")
            }
        finally:
            conn.close()
    except (sqlite3.Error, OSError) as error:
        logger.warning("agy attribution unreadable: %s: %s", store, error)
        return False, None
    _cache[str(store)] = (tuple(signature), agents)
    return True, agents.get(conversation)


class AgyAttributionGate:
    """Decides, per agy conversation, whether its turns may be indexed now and as which agent."""

    def __init__(self, state_path: str | Path):
        self.state_path = Path(state_path)
        self._alarmed: set[str] = set()
        self.state: dict[str, dict[str, Any]] = {"first_seen": {}, "fallback_indexed": {}, "late_memory_readers": {}}
        try:
            loaded = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            loaded = {}
        for key in self.state:
            if isinstance(loaded, dict) and isinstance(loaded.get(key), dict):
                self.state[key] = dict(loaded[key])

    def _prune(self) -> None:
        """Drop entries whose transcript is gone, and first_seen/fallback_indexed entries past STATE_MAX_AGE_S."""
        cutoff = _now() - STATE_MAX_AGE_S
        for section, ages in (("first_seen", True), ("fallback_indexed", True), ("late_memory_readers", False)):
            kept = {}
            for conversation, entry in self.state[section].items():
                if not isinstance(entry, dict):
                    continue
                path, at = entry.get("path"), entry.get("at")
                if not isinstance(path, str) or not os.path.exists(path):
                    continue
                if ages and (not isinstance(at, (int, float)) or at < cutoff):
                    continue
                kept[conversation] = entry
            self.state[section] = kept

    def save(self) -> None:
        """Prune, then replace the state file atomically (unique temp file, fsync, rename)."""
        self._prune()
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp = tempfile.mkstemp(dir=self.state_path.parent, prefix=f".{self.state_path.name}.")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.state, handle, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.state_path)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise

    def _entry(self, source_file: str | Path, **extra: Any) -> dict[str, Any]:
        return {"at": _now(), "path": os.path.abspath(os.path.expanduser(str(source_file))), **extra}

    def resolve(self, source_file: str | Path) -> tuple[bool, str | None]:
        """(index now, agent): agent is the row's agent_name, "" after the deadline, None while held."""
        located = agy_conversation(source_file)
        if located is None:
            return True, None
        store, conversation = located
        read_ok, agent = read_conversation_agent(source_file)
        if not read_ok or agent is None:
            # The deadline counts from the first unresolved sighting, readable or not.
            first_seen = self.state["first_seen"].get(conversation)
            if not isinstance(first_seen, dict):
                self.state["first_seen"][conversation] = first_seen = self._entry(source_file)
                self.save()
        if not read_ok:
            if str(store) not in self._alarmed:
                self._alarmed.add(str(store))
                try:
                    raise_alarm(
                        "agy_attribution_unreadable",
                        "agy conversations are held: their agent attribution could not be read",
                        {"store": str(store), "conversation": conversation},
                        severity="warning",
                    )
                except BrainLayerAlarm:
                    pass
            return False, None
        self._alarmed.discard(str(store))
        if agent is not None:
            changed = self.state["first_seen"].pop(conversation, None) is not None
            if (
                agent in MEMORY_READER_ATTRIBUTIONS
                and conversation in self.state["fallback_indexed"]
                and conversation not in self.state["late_memory_readers"]
            ):
                logger.warning(
                    "agy conversation %s was indexed after its attribution deadline and is now named "
                    "memory reader %r; no further turns are indexed, existing chunks are kept",
                    conversation,
                    agent,
                )
                self.state["late_memory_readers"][conversation] = self._entry(source_file, agent=agent)
                changed = True
            if changed:
                self.save()
            return True, agent
        if _now() - float(first_seen["at"]) < AGY_ATTRIBUTION_WAIT_S:
            return False, None
        if conversation not in self.state["fallback_indexed"]:
            self.state["fallback_indexed"][conversation] = self._entry(source_file)
            self.save()
        return True, ""
