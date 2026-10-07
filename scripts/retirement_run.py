"""Build and probe an exact source SHA; every failure leaves a report."""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.retirement_artifact import build_wheel, command, install_profile, private_env, snapshot  # noqa: E402
from scripts.retirement_inventory import check_policy  # noqa: E402
from scripts.retirement_policy_generate import render  # noqa: E402
from scripts.retirement_source import source_scan  # noqa: E402


def sandbox_command(args):
    if platform.system() != "Darwin" or not Path("/usr/bin/sandbox-exec").is_file():
        raise RuntimeError("Real macOS OS network boundary unavailable")
    return ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)", *args]


def resolve_dependency(repo, sha, ref, env):
    if not ref or ref.startswith("-"):
        raise ValueError("Dependency reference must be an explicit Git tag/base reference")
    base = subprocess.check_output(["git", "merge-base", sha, ref], cwd=repo, env=env, text=True).strip()
    if len(base) != 40 or any(c not in "0123456789abcdef" for c in base):
        raise ValueError("Dependency merge-base is not a full commit identity")
    return base


def probe_profile(profile, wheel, source_sha, dependency_sha, work, harness):
    python = profile["python"]
    env = private_env(Path(profile["home"]))
    env["PATH"] = str(Path(python).parent) + ":/usr/bin:/bin"
    site = Path(
        subprocess.check_output(
            [python, "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], env=env, text=True
        ).strip()
    )
    if not site.is_relative_to(Path(profile["prefix"])):
        raise ValueError("Guard installation outside private venv")
    destination = site / "sitecustomize.py"
    if destination.exists():
        raise ValueError("Preexisting early fixture")
    shutil.copyfile(harness / "retirement_guard.py", destination)
    events = work / (profile["profile"] + "-events.jsonl")
    env.update(
        BRAINLAYER_RETIREMENT_GUARD="1",
        RETIREMENT_EVENTS=str(events),
        RETIREMENT_PHASE="candidate",
        BRAINLAYER_READ_POOL_SIZE="1",
        RETIREMENT_OS_BOUNDARY="1",
        BRAINLAYER_ENRICHMENT_TAXONOMY_GIT_SHA=source_sha,
    )
    for key in (
        "GOOGLE_API_KEY",
        "GOOGLE_GENERATIVE_AI_API_KEY",
        "GROQ_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        env[key] = "SYNTHETIC_RETIREMENT_NOT_A_CREDENTIAL"
    env.update(
        BRAINLAYER_AUTO_ENRICH="1",
        BRAINLAYER_LLM_ENTITY_EXTRACTION="1",
        BRAINLAYER_ENRICH_BACKEND="gemini",
        BRAINLAYER_GROQ_BASE_URL="https://fixture.invalid",
        OLLAMA_HOST="https://fixture.invalid",
    )
    name = profile["profile"]
    config = {
        **profile,
        "source_sha": source_sha,
        "wheel_path": wheel["path"],
        "wheel_sha256": wheel["sha256"],
        "require_sdk_absence": bool(dependency_sha),
    }
    config_path = work / (name + "-input.json")
    output = work / (name + "-probe.json")
    config_path.write_text(json.dumps(config))
    # The CLI itself runs, not a source-tree python -m substitute.
    command(
        sandbox_command([str(Path(python).parent / "brainlayer"), "--help"]),
        Path(profile["home"]),
        env,
        work / (name + "-cli.log"),
        180,
    )
    cli = (work / (name + "-cli.log")).read_text()
    if "Usage:" not in cli or "Error in sitecustomize" in cli:
        raise ValueError("CLI/early fixture infrastructure failed")
    command(
        sandbox_command([python, "-I", str(harness / "retirement_probe.py"), str(config_path), str(output)]),
        Path(profile["home"]),
        env,
        work / (name + "-probe.log"),
        300,
    )
    result = json.loads(output.read_text())
    if result.get("status") != "PASS":
        raise ValueError("Installed probe failed")
    result.update(
        profile=name,
        source_sha=source_sha,
        cli="PASS",
        guard="private sitecustomize + OS deny network",
        wheel_sha256=wheel["sha256"],
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--sha", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--dependency-ref", help="Git tag/base reference; resolve its merge-base and require SDK absence"
    )
    args = parser.parse_args(argv)
    report = {
        "schema_version": 1,
        "scope": "no cloud model call reachable anywhere",
        "status": "FAIL",
        "source_sha": args.sha,
        "dependency_sha": None,
        "dependency_ref": args.dependency_ref,
        "dependency_resolution": "merge-base",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "profiles": [],
        "failures": [],
    }
    try:
        source_root, work = args.source.resolve(), args.work.resolve()
        if work.is_relative_to(source_root) or work.exists():
            raise ValueError("Use a fresh scratch work directory outside the checkout")
        work.mkdir(parents=True)
        if args.dependency_ref:
            report["dependency_sha"] = resolve_dependency(
                source_root, args.sha, args.dependency_ref, private_env(work / "build-home")
            )
            command(
                ["git", "merge-base", "--is-ancestor", report["dependency_sha"], args.sha],
                source_root,
                private_env(work / "build-home"),
                work / "dependency.log",
            )
            report["dependency_is_ancestor"] = True
        snapshot_root = work / "source"
        report["source"] = snapshot(source_root, args.sha, snapshot_root)
        # Execute only harness bytes included in this immutable source identity.
        for path in Path(__file__).parent.glob("retirement_*.py"):
            name = "scripts/" + path.name
            if hashlib.sha256(path.read_bytes()).hexdigest() != report["source"]["files"].get(name):
                raise ValueError("Harness differs from measured source: " + name)
        report["scan"] = source_scan(snapshot_root)
        if report["scan"]["findings"] or report["scan"]["errors"]:
            raise ValueError("Static model transport/source import gate RED")
        report["transport_inventory"] = check_policy(snapshot_root)
        if report["transport_inventory"]["unclassified"]:
            raise ValueError("Unclassified transport/dynamic source site")
        if (snapshot_root / "scripts/retirement_policy.json").read_text() != render(snapshot_root):
            raise ValueError("Closed transport policy is stale; explicit purpose review/regeneration required")
        report["wheel"] = build_wheel(snapshot_root, work, args.sha)
        for name in ("default", "dev"):
            profile = install_profile(Path(report["wheel"]["path"]), name, work)
            report["profiles"].append(
                probe_profile(
                    profile, report["wheel"], args.sha, report["dependency_sha"], work, Path(__file__).parent.resolve()
                )
            )
        from scripts.retirement_native import native_probe

        report["native"] = native_probe(snapshot_root, work)
        for filename, expected in report["source"]["files"].items():
            path = snapshot_root / filename
            data = str(path.readlink()).encode() if path.is_symlink() else path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("Build changed tracked source: " + filename)
        report["status"] = "PASS" if report["dependency_sha"] else "PENDING_R10B"
        if report["status"] == "PASS":
            from scripts.retirement_report import validate

            validate(report, args.sha)
    except Exception as error:
        report["status"] = "FAIL"
        report["failures"].append(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "source_sha", "dependency_sha", "failures")}))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
