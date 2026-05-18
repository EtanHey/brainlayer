"""Persistent BrainBar helper for canonical Python hybrid search.

BrainBar owns the MCP socket and write queue. This helper owns the Python
retrieval stack so BrainBar search results use the same RRF hybrid path as the
Python MCP implementation without porting ranking logic to Swift.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import sys
from pathlib import Path
from typing import Any


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


class HybridSearchHelper:
    def __init__(self, socket_path: Path, db_path: Path):
        self.socket_path = socket_path
        self.db_path = db_path
        self._stopped = False

    def warm(self) -> None:
        os.environ["BRAINLAYER_DB"] = os.fspath(self.db_path)
        from brainlayer.mcp._shared import _get_embedding_model, _get_vector_store

        _get_vector_store()
        model = _get_embedding_model()
        model.embed_query("brainbar hybrid helper warmup")

    def serve_forever(self) -> None:
        if self.socket_path.exists():
            self.socket_path.unlink()

        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            server.bind(os.fspath(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            server.listen(16)
            server.settimeout(0.25)
            self.warm()

            while not self._stopped:
                try:
                    conn, _ = server.accept()
                except TimeoutError:
                    continue
                except OSError:
                    if self._stopped:
                        break
                    raise
                with conn:
                    conn.settimeout(0.25)
                    self._handle_connection(conn)
        finally:
            server.close()
            try:
                self.socket_path.unlink()
            except FileNotFoundError:
                pass

    def stop(self, *_: object) -> None:
        self._stopped = True

    def _handle_connection(self, conn: socket.socket) -> None:
        try:
            raw = self._read_line(conn)
            request = json.loads(raw.decode("utf-8"))
            response = self._handle_request(request)
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}

        payload = json.dumps(_json_safe(response), separators=(",", ":")).encode("utf-8") + b"\n"
        conn.sendall(payload)

    @staticmethod
    def _read_line(conn: socket.socket) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = conn.recv(65536)
            if not chunk:
                break
            if b"\n" in chunk:
                before, _, _ = chunk.partition(b"\n")
                total += len(before)
                if total > 1_000_000:
                    raise ValueError("request exceeds 1MB")
                chunks.append(before)
                break
            total += len(chunk)
            if total > 1_000_000:
                raise ValueError("request exceeds 1MB")
            chunks.append(chunk)
        if not chunks:
            raise ValueError("empty request")
        return b"".join(chunks)

    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("method") != "brain_search":
            raise ValueError(f"unsupported method: {request.get('method')}")
        arguments = request.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be an object")

        text, structured = asyncio.run(self._search(arguments))
        metadata: dict[str, Any] = {}
        if structured is not None:
            metadata["structuredContent"] = structured
        return {"ok": True, "text": text, "metadata": metadata}

    async def _search(self, arguments: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        from brainlayer.mcp.search_handler import _brain_search

        source = arguments.get("source")
        if source is None or source == "":
            source = "all"

        result = await _brain_search(
            query=str(arguments.get("query") or ""),
            project=arguments.get("project"),
            source=source,
            tag=arguments.get("tag"),
            importance_min=arguments.get("importance_min"),
            num_results=int(arguments.get("num_results") or 5),
            detail=str(arguments.get("detail") or "compact"),
        )

        if isinstance(result, tuple):
            content, structured = result
            return self._content_text(content), structured if isinstance(structured, dict) else None
        if hasattr(result, "content"):
            return self._content_text(result.content), None
        return self._content_text(result), None

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, list):
            parts = []
            for item in content:
                text = getattr(item, "text", None)
                if text is None and isinstance(item, dict):
                    text = item.get("text")
                if text is not None:
                    parts.append(str(text))
            return "\n".join(parts)
        text = getattr(content, "text", None)
        if text is not None:
            return str(text)
        if isinstance(content, dict):
            text = content.get("text")
            if text is not None:
                return str(text)
        return str(content)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BrainBar persistent hybrid search helper")
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--db-path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.db_path:
        os.environ["BRAINLAYER_DB"] = args.db_path

    from brainlayer.paths import get_db_path

    helper = HybridSearchHelper(socket_path=Path(args.socket_path), db_path=get_db_path())
    signal.signal(signal.SIGTERM, helper.stop)
    signal.signal(signal.SIGINT, helper.stop)
    helper.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
