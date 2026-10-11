#!/usr/bin/env python3
"""Run the actual synthetic BrainBar views; missing pixels/OCR/builds fail closed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
from pathlib import Path

try:
    from scripts.brainbar_source_identity import source_identity
except ModuleNotFoundError:
    from brainbar_source_identity import source_identity

ROOT = Path(__file__).resolve().parents[1]
ICONS = {"status-icon-idle", "status-icon-active", "status-icon-badged"}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")


def digest_valid(value):
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


ROW = "BrainBar renders no enrichment status"
STATES = (
    "historical-274847",
    "empty",
    "loading",
    "stale",
    "error",
    "unavailable",
    "partial-replay",
    "draining",
    "backlogged",
)
CAPTURES = (
    ({"pending-store-replay"} | ICONS)
    | {f"dashboard-{state}" for state in STATES}
    | {f"{page}-{config}" for page in ("jobs", "backups", "advanced") for config in ("known", "unknown")}
)


def validate_report(payload: dict, expected_sha: str | None, expected_source: dict | None = None) -> str | None:
    if not expected_sha or payload.get("measured_sha") != expected_sha:
        return "render evidence does not belong to this checkout head"
    expected_source = expected_source or source_identity(ROOT)
    identity = payload.get("build_identity")
    if not isinstance(identity, dict) or identity.get("dirty") is not False or expected_source["dirty"]:
        return "missing clean compiled source identity"
    for key in ("schema_version", "head", "tree", "source_sha256", "source_files"):
        if identity.get(key) != expected_source[key]:
            return f"compiled source differs from checkout: {key}"
    if identity["head"] != expected_sha or not digest_valid(identity["source_sha256"]):
        return "compiled source digest/head malformed"
    if not isinstance(identity.get("root"), str) or not Path(identity["root"]).is_absolute():
        return "compiled source root missing"
    if type(payload.get("render_exit")) is not int or payload["render_exit"] != 0:
        return "successful measured render_exit missing"
    if payload.get("schema_version") != 1 or payload.get("row") != ROW or payload.get("status") != "PASS":
        return "actual-view render did not report PASS"
    if payload.get("mode") != "synthetic-source-build" or payload.get("historical_backlog") != 274847:
        return "adversarial historical-backlog render evidence missing"
    if payload.get("violations") != [] or payload.get("active_pending_store_visible") is not True:
        return "enrichment visible or active pending-store/replay UI missing"
    captures = payload.get("captures")
    if not isinstance(captures, list) or len(captures) != len(CAPTURES):
        return "incomplete actual-view capture inventory"
    if {item.get("name") for item in captures if isinstance(item, dict)} != CAPTURES:
        return "capture identity set differs from the required dashboard/settings states"
    for item in captures:
        if item.get("png") != item["name"] + ".png" or not digest_valid(item.get("png_sha256")):
            return "missing pixel artifact identity/digest"
        if item["name"] in ICONS:
            pixels = item.get("pixel_check", {})
            if not isinstance(pixels, dict) or pixels.get("series") != ["Agent", "Watcher"]:
                return "two-series icon proof missing"
            if pixels.get("matches_reference") is not True or not digest_valid(pixels.get("actual_rgba_sha256")):
                return "status icon pixels mismatch"
            if pixels.get("actual_rgba_sha256") != pixels.get("reference_rgba_sha256"):
                return "status icon reference digest differs"
            if type(pixels.get("nontransparent_pixels")) is not int or pixels["nontransparent_pixels"] <= 0:
                return "status icon invisible"
            if item["name"] == "status-icon-badged" and (
                type(pixels.get("red_pixels")) is not int or pixels["red_pixels"] <= 0
            ):
                return "attention badge missing"
            continue
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            return "missing OCR evidence"
        if "enrich" in text.lower() or "274,847" in text or "274847" in text:
            return f"{item['name']}: forbidden enrichment status/count/job text"
        marker = (
            ("loading" if item["name"] == "dashboard-loading" else "details")
            if item["name"].startswith("dashboard-")
            else ("pending stores" if item["name"] == "pending-store-replay" else "memory on this mac")
        )
        if marker not in text.lower():
            return f"{item['name']}: actual page proof marker missing"
        digest = item.get("png_sha256")
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            return "missing pixel artifact digest"
    control = payload.get("positive_control", {})
    if (
        not isinstance(control, dict)
        or control.get("name") != "ocr-positive-control"
        or control.get("png") != "ocr-positive-control.png"
        or not digest_valid(control.get("png_sha256"))
        or re.search(r"enrichment paused\s*[·•.]?\s*274,?847 queued", str(control.get("text", "")), re.I) is None
        or "enrichment retired" not in str(control.get("text", "")).lower()
    ):
        return "OCR forbidden-text positive control missing"
    active = next(item["text"].lower() for item in captures if item["name"] == "pending-store-replay")
    if "pending stores" not in active or "replay" not in active:
        return "active queue proof regions missing"
    digest = payload.get("binary_sha256", "")
    if not digest_valid(digest):
        return "exact binary identity missing"
    return None


def read_report(path: Path | None, expected_sha: str | None) -> tuple[dict | None, str | None]:
    if path is None:
        return None, "no macOS actual-view render evidence supplied"
    try:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            return None, "render report is not an object"
        return payload, validate_report(payload, expected_sha)
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        return None, f"render report unreadable: {type(error).__name__}"


def run(binary: Path, out: Path, root: Path, private_captures: Path | None = None) -> int:
    payload = {"schema_version": 1, "row": ROW, "status": "FAIL"}
    try:
        expected = source_identity(root)
        sha = expected["head"]
        payload["measured_sha"] = sha
        if expected["dirty"]:
            raise RuntimeError("dirty source checkout refused")
        if platform.system() != "Darwin" or not binary.is_file():
            raise RuntimeError("macOS and a built DEBUG BrainBar binary are required")
        out.parent.mkdir(parents=True, exist_ok=True)
        # Raw captures are private. CI uploads only this synthetic JSON receipt.
        with tempfile.TemporaryDirectory(prefix="brainbar-no-enrichment-") as temporary:
            private = Path(temporary)
            home = private / "home"
            home.mkdir(mode=0o700)
            env = dict(os.environ)
            env.update(
                HOME=str(home),
                CFFIXED_USER_HOME=str(home),
                BRAINLAYER_DB=str(private / "never-open.db"),
                BRAINBAR_SOCKET_PATH=str(private / "never-connect.sock"),
                BRAINLAYER_BRAINBAR_SOCKET=str(private / "never-connect.sock"),
                BRAINBAR_NO_ENRICHMENT_RENDER=str(private / "renders"),
                BRAINBAR_BUILD_IDENTITY_ONLY="1",
            )
            identity_result = subprocess.run(
                [str(binary.resolve())], cwd=root, env=env, capture_output=True, text=True, timeout=240
            )
            # Both flags are set: an older executable safely enters synthetic render
            # mode rather than starting the production app when it lacks identity mode.
            try:
                compiled = json.loads(identity_result.stdout)
            except ValueError as error:
                raise RuntimeError("executable lacks compiled source identity") from error
            if identity_result.returncode != 0 or compiled != expected:
                raise RuntimeError("executable build source differs from requested clean checkout")
            env.pop("BRAINBAR_BUILD_IDENTITY_ONLY")
            result = subprocess.run(
                [str(binary.resolve())], cwd=root, env=env, capture_output=True, text=True, timeout=240
            )
            print(result.stdout)
            print(result.stderr)
            report = private / "renders/render-report.json"
            if not report.is_file():
                raise RuntimeError(f"render exited {result.returncode} without a receipt")
            payload = json.loads(report.read_text())
            if payload.get("build_identity") != compiled or source_identity(root) != expected:
                raise RuntimeError("source/executable identity changed during render")
            if private_captures:
                private_captures.mkdir(mode=0o700, parents=True, exist_ok=True)
                if private_captures.stat().st_mode & 0o077:
                    raise RuntimeError("raw capture directory must be owner-only")
            for capture in [*payload["captures"], payload["positive_control"]]:
                if capture["png"] != capture["name"] + ".png" or Path(capture["png"]).name != capture["png"]:
                    raise RuntimeError("invalid capture filename")
                image = private / "renders" / capture["png"]
                image_bytes = image.read_bytes()
                capture["png_sha256"] = hashlib.sha256(image_bytes).hexdigest()
                if private_captures:
                    target = private_captures / capture["png"]
                    target.write_bytes(image_bytes)
                    target.chmod(0o600)
            payload["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
            payload["render_exit"] = result.returncode
            problem = validate_report(payload, sha, expected)
            if result.returncode != 0 or problem:
                raise RuntimeError(problem or f"render exited {result.returncode}")
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        payload["status"] = "FAIL"
        payload["reason"] = str(error)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0 if payload["status"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--private-captures", type=Path, help="Owner-only raw evidence directory; never upload")
    args = parser.parse_args()
    return run(args.binary, args.out, args.root, args.private_captures)


if __name__ == "__main__":
    raise SystemExit(main())
