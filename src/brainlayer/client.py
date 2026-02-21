"""Client for communicating with brainlayer daemon."""

import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

SOCKET_PATH = Path("/tmp/brainlayer.sock")
DAEMON_STARTUP_TIMEOUT = 30  # seconds


class DaemonClient:
    """Client for brainlayer daemon."""

    def __init__(self):
        self.base_url = "http://localhost"
        self._client: Optional[httpx.Client] = None

    def _get_client(self) -> httpx.Client:
        """Get HTTP client with Unix socket transport."""
        if self._client is None:
            transport = httpx.HTTPTransport(uds=str(SOCKET_PATH))
            self._client = httpx.Client(base_url=self.base_url, transport=transport, timeout=30.0)
        return self._client

    def _is_daemon_running(self) -> bool:
        """Check if daemon is running."""
        if not SOCKET_PATH.exists():
            return False

        try:
            client = self._get_client()
            response = client.get("/health")
            return response.status_code == 200
        except Exception:
            return False

    def _start_daemon(self) -> bool:
        """Start daemon process."""
        try:
            # Start daemon in background
            subprocess.Popen(
                ["brainlayer-daemon"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )

            # Wait for daemon to start
            for _ in range(DAEMON_STARTUP_TIMEOUT):
                if self._is_daemon_running():
                    return True
                time.sleep(1)

            return False

        except Exception as e:
            logger.warning("Failed to start daemon: %s", e)
            return False

    def _ensure_daemon(self) -> bool:
        """Ensure daemon is running."""
        if self._is_daemon_running():
            return True

        logger.info("Starting brainlayer daemon...")
        return self._start_daemon()

    def search(
        self,
        query: str,
        n_results: int = 10,
        project_filter: Optional[str] = None,
        content_type_filter: Optional[str] = None,
        source_filter: Optional[str] = None,
        use_semantic: bool = True,
        hybrid: bool = True,
    ) -> Dict[str, Any]:
        """Search the knowledge base."""
        if not self._ensure_daemon():
            raise RuntimeError("Failed to start daemon")

        try:
            client = self._get_client()
            response = client.post(
                "/search",
                json={
                    "query": query,
                    "n_results": n_results,
                    "project_filter": project_filter,
                    "content_type_filter": content_type_filter,
                    "source_filter": source_filter,
                    "use_semantic": use_semantic,
                    "hybrid": hybrid,
                },
            )
            response.raise_for_status()
            return response.json()

        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to communicate with daemon: {e}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Search failed: {e.response.text}")

    def get_context(self, chunk_id: str, before: int = 3, after: int = 3) -> Dict[str, Any]:
        """Get surrounding conversation context for a chunk."""
        if not self._ensure_daemon():
            raise RuntimeError("Failed to start daemon")

        try:
            client = self._get_client()
            response = client.get(f"/context/{chunk_id}", params={"before": before, "after": after})
            response.raise_for_status()
            return response.json()

        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to communicate with daemon: {e}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Context request failed: {e.response.text}")

    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self._ensure_daemon():
            raise RuntimeError("Failed to start daemon")

        try:
            client = self._get_client()
            response = client.get("/stats")
            response.raise_for_status()
            return response.json()

        except httpx.RequestError as e:
            raise RuntimeError(f"Failed to communicate with daemon: {e}")
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Stats request failed: {e.response.text}")

    def close(self):
        """Close client connection."""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Global client instance
_client: Optional[DaemonClient] = None


def get_client() -> DaemonClient:
    """Get global daemon client."""
    global _client
    if _client is None:
        _client = DaemonClient()
    return _client
