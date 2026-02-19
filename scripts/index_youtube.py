#!/usr/bin/env python3
"""Index YouTube video transcripts into BrainLayer.

Usage:
    # Single video
    python3 index_youtube.py https://www.youtube.com/watch?v=VIDEO_ID

    # Full channel (all videos)
    python3 index_youtube.py --channel UC2D2CMWXMOVWx7giW1n3LIg

    # Playlist
    python3 index_youtube.py --playlist PLZBnGBrX0YPn1Z_HqVG1zvJ0UqV

    # Dry run (show what would be indexed)
    python3 index_youtube.py --channel UC2D2CMWXMOVWx7giW1n3LIg --dry-run

    # Resume (skip already-indexed videos)
    python3 index_youtube.py --channel UC2D2CMWXMOVWx7giW1n3LIg --resume
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi

# Add brainlayer to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brainlayer.embeddings import embed_chunks
from brainlayer.paths import DEFAULT_DB_PATH
from brainlayer.pipeline.chunk import Chunk
from brainlayer.pipeline.classify import ContentType, ContentValue
from brainlayer.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DELAY_BETWEEN_VIDEOS = 10  # seconds, to avoid rate limiting
TARGET_CHUNK_CHARS = 1200  # ~300-400 tokens for bge-large (512 token limit)
CHUNK_OVERLAP_CHARS = 200


# ---------------------------------------------------------------------------
# Transcript extraction via youtube-transcript-api (primary — avoids yt-dlp 429s)
# ---------------------------------------------------------------------------

_yt_api = YouTubeTranscriptApi()  # Reuse single instance across calls


def get_transcript_via_api(video_id: str) -> list[dict] | None:
    """Fetch transcript using youtube-transcript-api (v1.2.4+).

    Uses a different internal YouTube endpoint than yt-dlp,
    so it often works when yt-dlp subtitle downloads are 429'd.
    Returns list of {"text": str, "start": float, "duration": float}.
    """
    try:
        transcript_list = _yt_api.list(video_id)
        transcript = None
        try:
            transcript = transcript_list.find_manually_created_transcript(["en"])
        except Exception:
            try:
                transcript = transcript_list.find_generated_transcript(["en"])
            except Exception:
                pass

        if not transcript:
            return None

        fetched = transcript.fetch()
        segments = []
        for entry in fetched:
            text = entry.text.strip()
            if text and text not in ("[Music]", "[Applause]", "[Laughter]"):
                segments.append({
                    "text": text,
                    "start": entry.start,
                    "duration": entry.duration,
                })
        return segments if segments else None
    except Exception as e:
        log.warning(f"  youtube-transcript-api failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Transcript extraction via yt-dlp (fallback)
# ---------------------------------------------------------------------------

def extract_video_info(video_url: str) -> dict[str, Any] | None:
    """Extract metadata + subtitles for a single video.

    Uses Brave browser cookies to bypass YouTube IP bans on subtitle requests.
    Uses process=False to skip format selection (avoids 'format not available' errors).
    """
    opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": ["en", "en-orig", "en.*"],
        "quiet": True,
        "no_warnings": True,
        "cookiesfrombrowser": ("brave",),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            return ydl.extract_info(video_url, download=False, process=False)
    except Exception as e:
        log.warning(f"Failed to extract {video_url}: {e}")
        return None


def get_transcript_via_download(video_url: str) -> list[dict] | None:
    """Download subtitles via yt-dlp and parse locally.

    Uses yt-dlp's built-in download mechanism which handles rate limits
    better than fetching subtitle URLs directly.
    """
    import tempfile
    import glob as glob_mod

    with tempfile.TemporaryDirectory() as tmpdir:
        opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "en-orig"],
            "subtitlesformat": "json3/vtt/srt",
            "outtmpl": f"{tmpdir}/%(id)s.%(ext)s",
            "quiet": True,
            "no_warnings": True,
            "cookiesfrombrowser": ("brave",),
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([video_url])
        except Exception as e:
            log.warning(f"  yt-dlp subtitle download failed: {e}")
            return None

        # Find downloaded subtitle files
        sub_files = glob_mod.glob(f"{tmpdir}/*.json3") + glob_mod.glob(f"{tmpdir}/*.vtt")
        if not sub_files:
            return None

        # Prefer json3
        json3_files = [f for f in sub_files if f.endswith(".json3")]
        vtt_files = [f for f in sub_files if f.endswith(".vtt")]

        if json3_files:
            return _parse_json3_file(json3_files[0])
        if vtt_files:
            return _parse_vtt_file(vtt_files[0])

    return None


def get_transcript_from_info(info: dict) -> list[dict] | None:
    """Extract transcript segments from yt-dlp info dict.

    Returns list of {"text": str, "start": float, "duration": float}.
    Tries manual subs first, falls back to auto-generated.
    """
    # Check for subtitles (manual first, then auto)
    for sub_key in ("subtitles", "automatic_captions"):
        subs = info.get(sub_key, {})
        # Try English variants
        for lang in ("en", "en-orig", "en-US", "en-GB"):
            if lang not in subs:
                continue
            formats = subs[lang]
            # Prefer json3 format (has word-level timing)
            for fmt in formats:
                if fmt.get("ext") == "json3":
                    return _fetch_json3_transcript(fmt["url"])
            # Fall back to vtt
            for fmt in formats:
                if fmt.get("ext") == "vtt":
                    return _fetch_vtt_transcript(fmt["url"])
    return None


def _get_ssl_context():
    """Get SSL context with proper certificates (fixes macOS cert issues)."""
    import ssl
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def _parse_json3_file(path: str) -> list[dict] | None:
    """Parse a json3 subtitle file from disk."""
    try:
        with open(path) as f:
            data = json.load(f)
        segments = []
        for event in data.get("events", []):
            if "segs" not in event:
                continue
            text = "".join(s.get("utf8", "") for s in event["segs"]).strip()
            if text and text not in ("[Music]", "[Applause]", "[Laughter]"):
                segments.append({
                    "text": text,
                    "start": event.get("tStartMs", 0) / 1000.0,
                    "duration": event.get("dDurationMs", 0) / 1000.0,
                })
        return segments if segments else None
    except Exception as e:
        log.warning(f"json3 file parse failed: {e}")
        return None


def _parse_vtt_file(path: str) -> list[dict] | None:
    """Parse a VTT subtitle file from disk."""
    try:
        with open(path) as f:
            vtt_text = f.read()
        return _parse_vtt_text(vtt_text)
    except Exception as e:
        log.warning(f"VTT file parse failed: {e}")
        return None


def _parse_vtt_text(vtt_text: str) -> list[dict] | None:
    """Parse VTT text into segments."""
    segments = []
    lines = vtt_text.split("\n")
    i = 0
    while i < len(lines):
        match = re.match(
            r"(\d{2}):(\d{2}):(\d{2})[.,](\d{3})\s*-->\s*(\d{2}):(\d{2}):(\d{2})[.,](\d{3})",
            lines[i].strip()
        )
        if match:
            h, m, s, ms = int(match[1]), int(match[2]), int(match[3]), int(match[4])
            start = h * 3600 + m * 60 + s + ms / 1000
            h2, m2, s2, ms2 = int(match[5]), int(match[6]), int(match[7]), int(match[8])
            end = h2 * 3600 + m2 * 60 + s2 + ms2 / 1000
            i += 1
            text_lines = []
            while i < len(lines) and lines[i].strip():
                line = lines[i].strip()
                line = re.sub(r"<[^>]+>", "", line)
                if line and line not in ("[Music]", "[Applause]", "[Laughter]"):
                    text_lines.append(line)
                i += 1
            text = " ".join(text_lines)
            if text:
                segments.append({
                    "text": text,
                    "start": start,
                    "duration": end - start,
                })
        i += 1
    return segments if segments else None


def _fetch_json3_transcript(url: str) -> list[dict] | None:
    """Fetch and parse json3 subtitle format (URL-based fallback)."""
    import urllib.request
    try:
        ctx = _get_ssl_context()
        with urllib.request.urlopen(url, timeout=30, context=ctx) as resp:
            data = json.loads(resp.read())
        segments = []
        for event in data.get("events", []):
            if "segs" not in event:
                continue
            text = "".join(s.get("utf8", "") for s in event["segs"]).strip()
            if text and text not in ("[Music]", "[Applause]", "[Laughter]"):
                segments.append({
                    "text": text,
                    "start": event.get("tStartMs", 0) / 1000.0,
                    "duration": event.get("dDurationMs", 0) / 1000.0,
                })
        return segments if segments else None
    except Exception as e:
        log.warning(f"json3 fetch failed: {e}")
        return None


def _fetch_vtt_transcript(url: str) -> list[dict] | None:
    """Fetch and parse VTT subtitle format (URL-based fallback)."""
    import urllib.request
    try:
        ctx = _get_ssl_context()
        with urllib.request.urlopen(url, timeout=30, context=ctx) as resp:
            vtt_text = resp.read().decode("utf-8")
        return _parse_vtt_text(vtt_text)
    except Exception as e:
        log.warning(f"VTT fetch failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Channel / playlist listing
# ---------------------------------------------------------------------------

def list_channel_videos(channel_id: str) -> list[dict]:
    """Get all video URLs + basic metadata from a channel."""
    url = f"https://www.youtube.com/channel/{channel_id}/videos"
    opts = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
        "cookiesfrombrowser": ("brave",),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            result = ydl.extract_info(url, download=False)
        entries = result.get("entries", [])
        videos = []
        for entry in entries:
            if entry and entry.get("id"):
                videos.append({
                    "id": entry["id"],
                    "title": entry.get("title", "Unknown"),
                    "url": f"https://www.youtube.com/watch?v={entry['id']}",
                })
        return videos
    except Exception as e:
        log.error(f"Failed to list channel {channel_id}: {e}")
        return []


def list_playlist_videos(playlist_id: str) -> list[dict]:
    """Get all video URLs from a playlist."""
    url = f"https://www.youtube.com/playlist?list={playlist_id}"
    opts = {
        "extract_flat": True,
        "quiet": True,
        "no_warnings": True,
        "cookiesfrombrowser": ("brave",),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            result = ydl.extract_info(url, download=False)
        entries = result.get("entries", [])
        videos = []
        for entry in entries:
            if entry and entry.get("id"):
                videos.append({
                    "id": entry["id"],
                    "title": entry.get("title", "Unknown"),
                    "url": f"https://www.youtube.com/watch?v={entry['id']}",
                })
        return videos
    except Exception as e:
        log.error(f"Failed to list playlist {playlist_id}: {e}")
        return []


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_transcript(
    segments: list[dict],
    video_id: str,
    title: str,
    channel: str,
    chapters: list[dict] | None = None,
    project: str = "huberman",
) -> list[Chunk]:
    """Chunk transcript segments into BrainLayer-compatible Chunk objects.

    If chapters are available, splits by chapter first, then by size.
    Otherwise uses sliding window with overlap.
    """
    if chapters:
        return _chunk_by_chapters(segments, video_id, title, channel, chapters, project)
    return _chunk_by_size(segments, video_id, title, channel, project)


def _chunk_by_chapters(
    segments: list[dict],
    video_id: str,
    title: str,
    channel: str,
    chapters: list[dict],
    project: str,
) -> list[Chunk]:
    """Split transcript by YouTube chapters, then sub-chunk large chapters."""
    chunks = []
    for chapter in chapters:
        ch_start = chapter.get("start_time", 0)
        ch_end = chapter.get("end_time", float("inf"))
        ch_title = chapter.get("title", "")

        # Gather segments in this chapter
        chapter_segs = [
            s for s in segments
            if ch_start <= s["start"] < ch_end
        ]
        if not chapter_segs:
            continue

        # Sub-chunk within the chapter
        sub_chunks = _sliding_window(chapter_segs, video_id, title, channel, project, ch_title)
        chunks.extend(sub_chunks)

    return chunks


def _chunk_by_size(
    segments: list[dict],
    video_id: str,
    title: str,
    channel: str,
    project: str,
) -> list[Chunk]:
    """Chunk by sliding window with overlap."""
    return _sliding_window(segments, video_id, title, channel, project, chapter_title=None)


def _sliding_window(
    segments: list[dict],
    video_id: str,
    title: str,
    channel: str,
    project: str,
    chapter_title: str | None,
) -> list[Chunk]:
    """Create chunks from segments using sliding window."""
    chunks = []
    current_texts: list[str] = []
    current_start: float | None = None
    current_end: float = 0
    current_chars = 0

    def flush():
        nonlocal current_texts, current_start, current_end, current_chars
        if not current_texts:
            return
        content = " ".join(current_texts)
        meta: dict[str, Any] = {
            "source": "youtube",
            "video_id": video_id,
            "title": title,
            "channel": channel,
            "start_seconds": int(current_start or 0),
            "end_seconds": int(current_end),
        }
        if chapter_title:
            meta["chapter"] = chapter_title
        chunks.append(Chunk(
            content=content,
            content_type=ContentType.ASSISTANT_TEXT,
            value=ContentValue.MEDIUM,
            metadata=meta,
            char_count=len(content),
        ))

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        if current_start is None:
            current_start = seg["start"]
        current_end = seg["start"] + seg.get("duration", 0)
        current_texts.append(text)
        current_chars += len(text) + 1  # +1 for space

        if current_chars >= TARGET_CHUNK_CHARS:
            flush()
            # Overlap: keep last ~CHUNK_OVERLAP_CHARS worth of text
            overlap_texts: list[str] = []
            overlap_chars = 0
            for t in reversed(current_texts):
                if overlap_chars + len(t) > CHUNK_OVERLAP_CHARS:
                    break
                overlap_texts.insert(0, t)
                overlap_chars += len(t) + 1
            current_texts = overlap_texts
            current_start = current_end - 10  # approximate overlap start
            current_chars = overlap_chars

    # Flush remainder
    flush()
    return chunks


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

def get_indexed_video_ids(store: VectorStore) -> set[str]:
    """Get set of video IDs already in the DB."""
    cursor = store.conn.cursor()
    rows = list(cursor.execute(
        "SELECT DISTINCT source_file FROM chunks WHERE source = 'youtube'"
    ))
    ids = set()
    for row in rows:
        sf = row[0]
        if sf.startswith("youtube:"):
            ids.add(sf.split(":", 1)[1])
    return ids


def index_single_video(
    video_url: str,
    store: VectorStore,
    project: str = "huberman",
    dry_run: bool = False,
) -> int:
    """Index a single YouTube video. Returns number of chunks indexed."""
    log.info(f"Extracting: {video_url}")
    info = extract_video_info(video_url)
    if not info:
        log.warning(f"Could not extract info for {video_url}")
        return 0

    video_id = info.get("id", "unknown")
    title = info.get("title", "Unknown")
    channel = info.get("uploader", info.get("channel", "Unknown"))
    chapters = info.get("chapters") or []
    upload_date = info.get("upload_date", "")

    log.info(f"  Title: {title}")
    log.info(f"  Chapters: {len(chapters)}")

    # Get transcript — try youtube-transcript-api first (different endpoint,
    # avoids yt-dlp 429s), then yt-dlp download, then URL-based extraction
    segments = get_transcript_via_api(video_id)
    if not segments:
        segments = get_transcript_via_download(video_url)
    if not segments:
        segments = get_transcript_from_info(info)
    if not segments:
        log.warning(f"  No transcript available for {video_id}")
        return 0

    log.info(f"  Transcript segments: {len(segments)}")

    # Chunk
    chunks = chunk_transcript(segments, video_id, title, channel, chapters, project)
    if not chunks:
        log.warning(f"  No chunks generated for {video_id}")
        return 0

    log.info(f"  Chunks: {len(chunks)}")

    if dry_run:
        log.info(f"  [DRY RUN] Would index {len(chunks)} chunks")
        for i, c in enumerate(chunks[:3]):
            log.info(f"    Chunk {i}: {c.content[:80]}...")
        return len(chunks)

    # Embed
    log.info(f"  Embedding {len(chunks)} chunks...")
    embedded = embed_chunks(chunks)

    # Build chunk_data for upsert
    source_file = f"youtube:{video_id}"
    chunk_data = []
    embeddings = []

    for i, ec in enumerate(embedded):
        c = ec.chunk
        # Add upload_date to metadata
        meta = dict(c.metadata)
        if upload_date:
            meta["upload_date"] = upload_date

        # Convert upload_date (YYYYMMDD) to ISO timestamp for created_at
        created_at = None
        if upload_date and len(upload_date) == 8:
            created_at = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}T00:00:00+00:00"

        chunk_data.append({
            "id": f"{source_file}:{i}",
            "content": c.content,
            "metadata": meta,
            "source_file": source_file,
            "project": project,
            "content_type": c.content_type.value,
            "value_type": c.value.value,
            "char_count": c.char_count,
            "source": "youtube",
            "conversation_id": source_file,
            "position": i,
            "created_at": created_at,
        })
        embeddings.append(ec.embedding)

    # Upsert
    n = store.upsert_chunks(chunk_data, embeddings)
    log.info(f"  Indexed {n} chunks for '{title}'")
    return n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Index YouTube transcripts into BrainLayer"
    )
    parser.add_argument("url", nargs="?", help="Single video URL")
    parser.add_argument("--channel", help="YouTube channel ID (index all videos)")
    parser.add_argument("--playlist", help="YouTube playlist ID")
    parser.add_argument("--project", default="huberman", help="BrainLayer project name")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="BrainLayer DB path")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be indexed")
    parser.add_argument("--resume", action="store_true", help="Skip already-indexed videos")
    parser.add_argument("--delay", type=float, default=DELAY_BETWEEN_VIDEOS,
                        help="Delay between videos (seconds)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max videos to process (0=all)")
    args = parser.parse_args()

    if not args.url and not args.channel and not args.playlist:
        parser.error("Provide a video URL, --channel, or --playlist")

    store = VectorStore(args.db)
    indexed_ids = get_indexed_video_ids(store) if args.resume else set()

    if args.resume:
        log.info(f"Resume mode: {len(indexed_ids)} videos already indexed")

    total_chunks = 0
    total_videos = 0
    skipped = 0
    failed = 0

    if args.url:
        # Single video
        total_chunks = index_single_video(args.url, store, args.project, args.dry_run)
        total_videos = 1 if total_chunks > 0 else 0
    else:
        # Channel or playlist
        if args.channel:
            log.info(f"Listing videos for channel {args.channel}...")
            videos = list_channel_videos(args.channel)
        else:
            log.info(f"Listing videos for playlist {args.playlist}...")
            videos = list_playlist_videos(args.playlist)

        log.info(f"Found {len(videos)} videos")

        if args.limit > 0:
            videos = videos[:args.limit]
            log.info(f"Limiting to {args.limit} videos")

        for i, video in enumerate(videos):
            vid = video["id"]

            if vid in indexed_ids:
                log.info(f"[{i+1}/{len(videos)}] SKIP (already indexed): {video['title']}")
                skipped += 1
                continue

            log.info(f"[{i+1}/{len(videos)}] Processing: {video['title']}")
            try:
                n = index_single_video(video["url"], store, args.project, args.dry_run)
                if n > 0:
                    total_chunks += n
                    total_videos += 1
                else:
                    failed += 1
            except Exception as e:
                log.error(f"  FAILED: {e}")
                failed += 1

            # Rate limit delay
            if i < len(videos) - 1:
                time.sleep(args.delay)

    store.close()

    log.info("=" * 60)
    log.info(f"Done! Videos indexed: {total_videos}, chunks: {total_chunks}")
    if skipped:
        log.info(f"Skipped (already indexed): {skipped}")
    if failed:
        log.info(f"Failed: {failed}")


if __name__ == "__main__":
    main()
