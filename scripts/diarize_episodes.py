#!/usr/bin/env python3
"""Diarize guest episodes and re-index into BrainLayer.

Pipeline per episode:
  1. Download audio via yt-dlp (wav format)
  2. Transcribe + align + diarize via WhisperX
  3. Map SPEAKER_00/01 → "Huberman"/"Guest" (by speaking time)
  4. Save diarized JSON
  5. Re-index via index_youtube.py --diarized-transcript --replace

Usage:
    # All guest episodes
    python3 diarize_episodes.py

    # Single episode
    python3 diarize_episodes.py --video-id zEYE-vcVKy8

    # Dry run (download + diarize only, don't re-index)
    python3 diarize_episodes.py --dry-run

    # Skip download (reuse existing audio)
    python3 diarize_episodes.py --skip-download

Prerequisites:
    - WhisperX installed: pip install git+https://github.com/m-bain/whisperx.git
    - HuggingFace token at ~/.huggingface/token
    - pyannote model terms accepted at huggingface.co/pyannote/speaker-diarization-3.1
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Guest episodes that need diarization
EPISODES = [
    {"video_id": "zEYE-vcVKy8", "title": "Galpin: Assess & Improve Fitness", "guest": "Galpin"},
    {"video_id": "CyDLbrZK75U", "title": "Galpin: Build Strength & Muscles", "guest": "Galpin"},
    {"video_id": "oNkDA2F7CjM", "title": "Galpin: Endurance & Fat Loss", "guest": "Galpin"},
    {"video_id": "UIy-WQCZd4M", "title": "Galpin: Training Program Design", "guest": "Galpin"},
    {"video_id": "juD99_sPWGU", "title": "Galpin: Recovery", "guest": "Galpin"},
    {"video_id": "q37ARYnRDGc", "title": "Galpin: Nutrition & Supplements", "guest": "Galpin"},
    {"video_id": "IAnhFUUCq6c", "title": "Galpin original #65", "guest": "Galpin"},
    {"video_id": "x3MgDtZovks", "title": "Søberg Cold/Heat", "guest": "Søberg"},
]

AUDIO_DIR = Path(__file__).parent.parent / "data" / "audio"
DIARIZED_DIR = Path(__file__).parent.parent / "data" / "diarized"


def get_hf_token() -> str:
    """Read HuggingFace token from standard location."""
    token_path = Path.home() / ".huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    token = os.environ.get("HF_TOKEN", "")
    if token:
        return token
    raise RuntimeError(
        "No HuggingFace token found. Create one at https://huggingface.co/settings/tokens "
        "and save to ~/.huggingface/token"
    )


def download_audio(video_id: str, output_dir: Path) -> Path:
    """Download audio from YouTube as WAV."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_id}.wav"

    if output_path.exists():
        log.info(f"  Audio already exists: {output_path}")
        return output_path

    log.info(f"  Downloading audio for {video_id}...")
    cmd = [
        "yt-dlp",
        "-x", "--audio-format", "wav",
        "--output", str(output_path),
        "--cookies-from-browser", "brave",
        "--remote-components", "ejs:github",
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"  yt-dlp failed: {result.stderr}")
        raise RuntimeError(f"Failed to download audio for {video_id}")

    # yt-dlp might add extra extension
    if not output_path.exists():
        # Check for .wav.wav or similar
        for f in output_dir.glob(f"{video_id}*"):
            if f.suffix in (".wav", ".mp3", ".m4a"):
                output_path = f
                break

    log.info(f"  Audio saved: {output_path}")
    return output_path


def diarize_audio(audio_path: Path, hf_token: str) -> list[dict]:
    """Run WhisperX transcribe + align + diarize on audio file.

    Caches aligned transcript to avoid re-transcribing on diarization failures.
    Returns list of segments with speaker labels:
    [{"start": 0.5, "end": 2.3, "speaker": "SPEAKER_00", "text": "..."}]
    """
    try:
        import whisperx
    except ImportError:
        raise RuntimeError(
            "WhisperX not installed. Run: pip install git+https://github.com/m-bain/whisperx.git"
        )

    import torch

    # CTranslate2 (faster-whisper) doesn't support MPS — use CPU for transcription
    # pyannote diarization also needs CPU (MPS support is partial)
    if torch.cuda.is_available():
        device = "cuda"
        compute_type = "float16"
    else:
        device = "cpu"
        compute_type = "int8"

    log.info(f"  Device: {device}, compute: {compute_type}")

    # Check for cached aligned transcript (avoids re-transcribing on diarization failures)
    cache_dir = audio_path.parent.parent / "data" / "aligned"
    cache_path = cache_dir / f"{audio_path.stem}_aligned.json"

    audio = whisperx.load_audio(str(audio_path))

    if cache_path.exists():
        log.info(f"  Using cached aligned transcript: {cache_path}")
        with open(cache_path) as f:
            result = json.load(f)
    else:
        # 1. Transcribe
        log.info("  Transcribing (this takes a while on CPU)...")
        model = whisperx.load_model("large-v3", device, compute_type=compute_type)
        result = model.transcribe(audio, batch_size=16 if device == "cuda" else 4)
        del model

        # 2. Align
        log.info("  Aligning...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], device=device
        )
        result = whisperx.align(
            result["segments"], model_a, metadata, audio, device,
            return_char_alignments=False,
        )
        del model_a

        # Cache aligned result
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(result, f)
        log.info(f"  Cached aligned transcript: {cache_path}")

    # 3. Diarize
    log.info("  Diarizing...")
    from whisperx.diarize import DiarizationPipeline
    diarize_model = DiarizationPipeline(
        token=hf_token, device="cpu"
    )
    diarize_segments = diarize_model(audio)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    del diarize_model
    if device == "cuda":
        torch.cuda.empty_cache()

    return result["segments"]


def _detect_huberman_speaker(segments: list[dict]) -> str | None:
    """Content-based detection: find which speaker ID says Huberman's intro.

    Scans early segments for self-identification phrases like
    "I'm Andrew Huberman" or "Huberman Lab". Returns the speaker ID
    that says these phrases, or None if not found.
    """
    # Only check first 60 seconds of content
    intro_phrases = [
        "i'm andrew huberman",
        "my name is andrew huberman",
        "i am andrew huberman",
        "huberman lab",
        "welcome to the huberman lab",
    ]
    for seg in segments:
        if seg.get("start", 0) > 120:  # Only check first 2 min
            break
        text = seg.get("text", "").lower()
        for phrase in intro_phrases:
            if phrase in text:
                speaker = seg.get("speaker")
                if speaker:
                    return speaker
    return None


def map_speakers(segments: list[dict], guest_name: str) -> list[dict]:
    """Map SPEAKER_00/01 to Huberman/Guest.

    Uses two strategies:
    1. Content-based: scan intro for "I'm Andrew Huberman" (most reliable)
    2. Speaking time fallback: Huberman talks more in his episodes
    """
    speaker_time: dict[str, float] = {}
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        duration = seg.get("end", 0) - seg.get("start", 0)
        speaker_time[speaker] = speaker_time.get(speaker, 0) + duration

    if not speaker_time:
        log.warning("  No speakers found in segments")
        return segments

    sorted_speakers = sorted(speaker_time.items(), key=lambda x: x[1], reverse=True)

    # Strategy 1: Content-based detection (preferred)
    huberman_id = _detect_huberman_speaker(segments)
    if huberman_id:
        log.info(f"  Content-based detection: Huberman = {huberman_id}")
    else:
        # Strategy 2: Most talkative = Huberman
        huberman_id = sorted_speakers[0][0]
        log.info(f"  Fallback to speaking-time: Huberman = {huberman_id}")

    speaker_map = {huberman_id: "Huberman"}
    for spk, _ in sorted_speakers:
        if spk == huberman_id:
            continue
        if spk not in speaker_map:
            if guest_name not in speaker_map.values():
                speaker_map[spk] = guest_name
            else:
                idx = len(speaker_map)
                speaker_map[spk] = f"Speaker_{idx}"

    log.info(f"  Speaker mapping: {speaker_map}")
    log.info(f"  Speaking time: {dict((speaker_map.get(k, k), f'{v:.0f}s') for k, v in speaker_time.items())}")

    # Apply mapping
    for seg in segments:
        old_speaker = seg.get("speaker", "UNKNOWN")
        seg["speaker"] = speaker_map.get(old_speaker, old_speaker)

    return segments


def reindex_episode(video_id: str, diarized_path: Path) -> int:
    """Re-index episode using index_youtube.py with diarized transcript."""
    script_path = Path(__file__).parent / "index_youtube.py"
    venv_python = Path(__file__).parent.parent / ".venv" / "bin" / "python3"

    cmd = [
        str(venv_python),
        str(script_path),
        f"https://www.youtube.com/watch?v={video_id}",
        "--diarized-transcript", str(diarized_path),
        "--replace",
    ]

    log.info(f"  Re-indexing with: {' '.join(cmd[-4:])}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"  Re-index failed: {result.stderr}")
        return 0

    # Parse chunk count from output
    for line in result.stdout.split("\n"):
        if "Indexed" in line and "chunks" in line:
            log.info(f"  {line.strip()}")

    return 1


def process_episode(
    episode: dict,
    hf_token: str,
    dry_run: bool = False,
    skip_download: bool = False,
) -> bool:
    """Process a single episode: download → diarize → map speakers → save → reindex."""
    video_id = episode["video_id"]
    guest = episode["guest"]
    title = episode["title"]

    log.info(f"\n{'='*60}")
    log.info(f"Processing: {title} ({video_id})")
    log.info(f"Guest: {guest}")

    # 1. Download audio
    if skip_download:
        audio_path = AUDIO_DIR / f"{video_id}.wav"
        if not audio_path.exists():
            log.error(f"  --skip-download but no audio at {audio_path}")
            return False
    else:
        try:
            audio_path = download_audio(video_id, AUDIO_DIR)
        except Exception as e:
            log.error(f"  Download failed: {e}")
            return False

    # 2. Diarize
    diarized_path = DIARIZED_DIR / f"{video_id}.json"
    if diarized_path.exists():
        log.info(f"  Diarized transcript already exists: {diarized_path}")
        with open(diarized_path) as f:
            segments = json.load(f)
    else:
        try:
            segments = diarize_audio(audio_path, hf_token)
        except Exception as e:
            log.error(f"  Diarization failed: {e}")
            return False

        # 3. Map speakers
        segments = map_speakers(segments, guest)

        # 4. Save diarized JSON
        DIARIZED_DIR.mkdir(parents=True, exist_ok=True)
        with open(diarized_path, "w") as f:
            json.dump(segments, f, indent=2)
        log.info(f"  Saved diarized transcript: {diarized_path} ({len(segments)} segments)")

    if dry_run:
        log.info("  [DRY RUN] Skipping re-index")
        # Show sample
        for seg in segments[:5]:
            speaker = seg.get("speaker", "?")
            text = seg.get("text", "")[:80]
            log.info(f"    {speaker}: {text}")
        return True

    # 5. Re-index
    try:
        reindex_episode(video_id, diarized_path)
    except Exception as e:
        log.error(f"  Re-index failed: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Diarize Huberman guest episodes and re-index in BrainLayer"
    )
    parser.add_argument("--video-id", help="Process single episode by video ID")
    parser.add_argument("--dry-run", action="store_true", help="Download + diarize only, skip re-index")
    parser.add_argument("--skip-download", action="store_true", help="Skip audio download (reuse existing)")
    parser.add_argument("--batch-size", type=int, default=3,
                        help="Max episodes per batch (default 3, for thermal safety)")
    args = parser.parse_args()

    # Get HF token
    try:
        hf_token = get_hf_token()
        log.info(f"HuggingFace token found ({len(hf_token)} chars)")
    except RuntimeError as e:
        log.error(str(e))
        sys.exit(1)

    # Select episodes
    if args.video_id:
        episodes = [e for e in EPISODES if e["video_id"] == args.video_id]
        if not episodes:
            log.error(f"Video ID {args.video_id} not in episode list")
            sys.exit(1)
    else:
        episodes = EPISODES[:args.batch_size]
        log.info(f"Processing batch of {len(episodes)} / {len(EPISODES)} episodes")

    # Process
    succeeded = 0
    failed = 0
    for ep in episodes:
        ok = process_episode(ep, hf_token, args.dry_run, args.skip_download)
        if ok:
            succeeded += 1
        else:
            failed += 1

    log.info(f"\n{'='*60}")
    log.info(f"Done! Succeeded: {succeeded}, Failed: {failed}")
    if len(EPISODES) > len(episodes):
        remaining = len(EPISODES) - len(episodes)
        log.info(f"Remaining episodes: {remaining}. Run again for next batch.")


if __name__ == "__main__":
    main()
