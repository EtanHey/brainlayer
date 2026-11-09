"""Explicit, bounded admission of immutable Codex archives to the existing writer."""

import json
import logging
import os
from contextlib import contextmanager
from pathlib import Path

from .watcher import JSONLTailer, _source_fingerprint
from .zstd_stream import ZstdDecodedFile

logger = logging.getLogger(__name__)


def zst_enabled() -> bool:
    mode = os.environ.get("BRAINLAYER_CODEX_ZST_ADMISSION", "off")
    if mode not in {"off", "bounded"}:
        raise ValueError("BRAINLAYER_CODEX_ZST_ADMISSION must be off or bounded")
    return mode == "bounded"


def zst_limits() -> tuple[int, int]:
    values = tuple(
        int(os.environ.get(key, default))
        for key, default in (
            ("BRAINLAYER_CODEX_ZST_FILES_PER_CYCLE", "2"),
            ("BRAINLAYER_CODEX_ZST_BYTES_PER_CYCLE", "1048576"),
        )
    )
    if min(values) <= 0:
        raise ValueError("Codex zst file and decoded-byte budgets must be positive")
    return values


def queue_has_capacity() -> bool:
    from .health_check import HealthCheckConfig
    from .queue_io import get_queue_dir

    # Reuse the existing writer's queue and backlog threshold; no competing DB writer.
    threshold = HealthCheckConfig().queue_auto_heal_count
    try:
        with os.scandir(get_queue_dir()) as entries:
            count = 0
            for entry in entries:
                if entry.name.endswith(".jsonl"):
                    count += 1
                    if count >= threshold:
                        return False
        return True
    except FileNotFoundError:
        return True
    except OSError:
        logger.exception("Holding Codex zst admission: queue depth unavailable")
        return False


class ZstdTailer(JSONLTailer):
    """Decoded offsets survive representation changes; a live decoder spans bounded polls.

    The compressed inode is not a logical generation: Codex publishes a new inode when
    compressing/materializing the SAME rollout. Stat tokens still gate immutable EOFs.
    """

    def __init__(self, filepath, offset=0, max_record_bytes=None, *, resume_fingerprint=None):
        super().__init__(filepath, offset, max_record_bytes, resume_fingerprint=resume_fingerprint)
        self._replay = None
        self._archive_stat = None
        self._verified_archive_stat = None
        self._expected_fingerprint = resume_fingerprint
        self._archive_reconcile_through = 0
        self.decoder = None
        self.capture_header = True
        self.session_identity = None
        self._header_bytes = bytearray()
        self._header_done = False
        self.decoded_eof = False
        self.bytes_read = 0
        self.terminal_failure = None

    def get_inode(self):
        return 0

    def check_rewind(self):
        return False

    def close(self):
        if self.decoder is not None:
            self.decoder.close()
            self.decoder = None
        self._replay = None

    def restore_snapshot(self, snapshot):
        self.close()
        self.decoded_eof = False
        super().restore_snapshot(snapshot)
        if self.anchor is None:
            self._resume_fingerprint = self._expected_fingerprint

    def _reconcile_from_zero(self, reason):
        logger.warning(
            "Archive checkpoint not verified: %s (offset %d: %s); replaying from decoded byte 0",
            self.filepath,
            self.offset,
            reason,
        )
        self._archive_reconcile_through = max(self._archive_reconcile_through, self.offset)
        self.offset = 0
        self._buffer = b""
        self._reset_identity()
        self._verified_archive_stat = None
        self.close()

    @contextmanager
    def _open_window(self, max_bytes):
        self.bytes_read = 0
        self.decoded_eof = False
        source_stat = self.source_stat()
        if self.decoder is None:
            if self.offset and self.anchor is None and self._resume_fingerprint is None:
                self._reconcile_from_zero("checkpoint has no fingerprint; archive mtime is not evidence")
            if hasattr(self, "_header_bytes"):
                self._header_bytes.clear()
                self._header_done = False
            self._archive_stat = source_stat
            self._replay = JSONLTailer(self.filepath)
            self.decoder = ZstdDecodedFile(Path(self.filepath))
        elif source_stat != self._archive_stat:
            self.close()
            raise OSError("compressed source changed during decoding")
        decoder = self.decoder
        start = decoder.position
        try:
            target = self.offset + len(self._buffer)
            remaining = max_bytes if max_bytes is not None else 1048576
            while decoder.position < target and remaining > 0:
                data = decoder.read(min(65536, target - decoder.position, remaining))
                if not data:
                    raise OSError("compressed rollout ends before its decoded checkpoint")
                position = decoder.position - len(data)
                self._capture_prefix_header(data)
                self._replay._consume_anchor(data[: max(0, self.offset - position)])
                buffered = data[max(0, self.offset - position) :]
                buffer_start = max(0, position - self.offset)
                if buffered != self._buffer[buffer_start : buffer_start + len(buffered)]:
                    self._reconcile_from_zero("buffered bytes changed")
                    yield None, 0
                    return
                remaining -= len(data)
            if decoder.position < target:
                yield None, 0
                return
            if self._replay is not None:
                expected = (
                    self._resume_fingerprint
                    if self.anchor is None
                    else _source_fingerprint(self.offset, self._head_hash.digest(), self.anchor)
                )
                actual = _source_fingerprint(self.offset, self._replay._head_hash.digest(), self._replay.anchor)
                if actual != expected:
                    self._reconcile_from_zero("persisted fingerprint does not match")
                    yield None, 0
                    return
                if self.source_stat() != source_stat:
                    raise OSError("compressed source changed during prefix verification")
                self._adopt_identity(self._replay._head_hash, self._replay._head_len, self._replay.anchor)
                self._verified_archive_stat = source_stat
                self._replay = None
            yield decoder, remaining
            if self.source_stat() != source_stat:
                raise OSError("compressed source changed during decoding")
            if decoder.eof:
                self.decoded_eof = True
                self.close()
        except BaseException:
            self.close()
            raise
        finally:
            self.bytes_read = decoder.position - start

    def _capture_prefix_header(self, data):
        if self._header_done:
            return
        from .t3_provenance import compact_codex_header

        end = data.find(b"\n")
        part = data if end < 0 else data[: end + 1]
        if len(self._header_bytes) + len(part) > 1048576:
            self.first_record = compact_codex_header({}, 1048577)
            self._header_done = True
        else:
            self._header_bytes.extend(part)
            if end < 0:
                return
            self._header_done = True
            try:
                record = json.loads(self._header_bytes)
            except (ValueError, UnicodeDecodeError):
                record = {}
            self.first_record = compact_codex_header(record, len(self._header_bytes))
            payload = record.get("payload") if isinstance(record, dict) else None
            if isinstance(record, dict) and record.get("type") == "session_meta" and isinstance(payload, dict):
                self.session_identity = (payload.get("id"), payload.get("cwd"))
        self._header_bytes.clear()

    def source_stat(self):
        st = Path(self.filepath).stat()
        return st.st_size, st.st_mtime_ns, st.st_ino

    def read_new_lines(self, max_lines=None, max_bytes=None):
        result = (
            self.read_buffered_lines(max_lines) if self.decoded_eof else super().read_new_lines(max_lines, max_bytes)
        )
        if self.decoded_eof and self._buffer and not self.has_complete_buffered_line():
            if max_lines is None or len(result) < max_lines:
                try:
                    parsed = json.loads(self._buffer)
                    if not isinstance(parsed, dict):
                        raise ValueError("final JSONL record is not an object")
                except (ValueError, UnicodeDecodeError) as error:
                    self.terminal_failure = "truncated_final_record"
                    self.last_error = ValueError(f"truncated_final_record: {error}")
                    self.failed_record = self._buffer
                else:
                    self._consume_anchor(self._buffer)
                    self.offset += len(self._buffer)
                    self._remember_fingerprint(self.offset)
                    self._buffer = b""
                    parsed["_line_end_offset"] = self.offset
                    result.append(parsed)
        return result
