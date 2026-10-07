"""Runs under private installed Python -I, with sitecustomize already armed."""

import asyncio
import builtins
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import zipfile
from pathlib import Path

import sitecustomize as guard


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def controls():
    os.environ["RETIREMENT_PHASE"] = "control"
    count = 0

    def blocked(call):
        nonlocal count
        try:
            call()
        except guard.Attempt:
            count += 1
        else:
            raise RuntimeError("Early guard positive control did not refuse")

    with socket.socket(socket.AF_INET) as sock:
        blocked(lambda: sock.connect(("127.0.0.1", 9)))
    with socket.socket(socket.AF_INET6) as sock:
        blocked(lambda: sock.connect_ex(("::1", 9, 0, 0)))
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        blocked(lambda: sock.sendto(b"fixture", ("127.0.0.1", 9)))
    blocked(lambda: socket.getaddrinfo("fixture.invalid", 80))
    family, child = guard.FORBIDDEN[0].rsplit(".", 1)
    blocked(lambda: builtins.__import__(family, fromlist=(child,)))
    blocked(
        lambda: builtins.__import__(
            guard.FORBIDDEN[-3].rsplit(".", 1)[1], {"__package__": "brainlayer"}, fromlist=("X",), level=1
        )
    )
    blocked(lambda: importlib.import_module(guard.FORBIDDEN[-1]))
    blocked(lambda: subprocess.Popen([sys.executable, "-c", "pass"]))
    requests = importlib.import_module("requests")
    sender = requests.Session().send
    blocked(lambda: sender(requests.Request("GET", "http://fixture.invalid").prepare()))
    httpx = importlib.import_module("httpx")
    blocked(lambda: httpx.Client().send(httpx.Request("GET", "http://fixture.invalid")))
    blocked(lambda: asyncio.run(httpx.AsyncClient().send(httpx.Request("GET", "http://fixture.invalid"))))
    thread = threading.Thread(target=lambda: blocked(lambda: socket.getaddrinfo("fixture.invalid", 80)))
    thread.start()
    thread.join(5)
    require(not thread.is_alive(), "Control thread abandoned")
    os.environ["RETIREMENT_PHASE"] = "candidate"
    return count


def inspect_sdk(require_absence):
    importable = []
    for name in (
        "google.genai",
        "google.generativeai",
        "google.cloud.aiplatform",
        "vertexai",
        "groq",
        "openai",
        "anthropic",
        "cohere",
        "mistralai",
        "litellm",
        "xai_sdk",
        "together",
    ):
        try:
            spec = importlib.util.find_spec(name)
        except ModuleNotFoundError as error:
            if error.name != name and not name.startswith(error.name + "."):
                raise
            spec = None
        if spec:
            importable.append(name)
    distribution = importlib.metadata.distribution("brainlayer")
    from packaging.requirements import Requirement

    names = {
        "google-genai",
        "google-generativeai",
        "groq",
        "openai",
        "anthropic",
        "cohere",
        "mistralai",
        "litellm",
        "xai-sdk",
        "together",
        "google-cloud-aiplatform",
    }
    declared = [
        value for value in distribution.requires or [] if Requirement(value).name.lower().replace("_", "-") in names
    ]
    if require_absence:
        require(not importable and not declared, "Model SDK installation/declaration survives R10b")
    return {"status": "PASS" if require_absence else "PENDING_R10B", "importable": importable, "declared": declared}


def run(config):
    require(guard.ARMED, "sitecustomize was not armed")
    require(sys.flags.isolated == 1 and not os.environ.get("PYTHONPATH"), "Python is not isolated")
    require(Path(sys.prefix).resolve() == Path(config["prefix"]).resolve(), "Wrong fixture interpreter")
    import brainlayer

    origin = Path(brainlayer.__file__).resolve()
    require(origin.is_relative_to(Path(sys.prefix).resolve()), "Wrong installed origin")
    require(brainlayer.__build_sha__ == config["source_sha"], "Installed stamp is not measured source")
    distribution = importlib.metadata.distribution("brainlayer")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    require(not direct.get("dir_info", {}).get("editable"), "Editable installation")
    wheel = Path(config["wheel_path"])
    require(hashlib.sha256(wheel.read_bytes()).hexdigest() == config["wheel_sha256"], "Wheel drift")
    modules, verified = [], 0
    with zipfile.ZipFile(wheel) as zipped:
        for name in zipped.namelist():
            if name.startswith("brainlayer/") and not name.endswith("/"):
                require(
                    Path(distribution.locate_file(name)).read_bytes() == zipped.read(name),
                    "Installed member differs: " + name,
                )
                verified += 1
                if name.endswith(".py"):
                    module = name[:-3].replace("/", ".").removesuffix(".__init__")
                    modules.append(module)
    require(verified > 0 and modules, "Empty installed package inventory")
    positive_count = controls()
    require(positive_count == 12, "Incomplete guard controls")
    for module in sorted(set(modules)):
        loaded = importlib.import_module(module)
        require(
            Path(loaded.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), "Import resolved to source tree"
        )
    from brainlayer.pipeline import SanitizeConfig, Sanitizer, build_external_prompt

    sanitizer = Sanitizer(SanitizeConfig(owner_names=("Person Alpha",), use_spacy_ner=False))
    prompt, _ = build_external_prompt(
        {"content": "Person Alpha fixed {bug}", "project": "fixture", "content_type": "user_message"},
        sanitizer,
        [],
        "{content}",
    )
    require("[OWNER]" in prompt, "Retained Sanitizer/public prompt contract failed")
    from brainlayer.pipeline.enrichment_results import parse_enrichment

    parsed = parse_enrichment('{"summary":"A synthetic historical decision", "tags":["fixture"]}')
    require(parsed and parsed["summary"] == "A synthetic historical decision", "Saved-result parser failed")
    from brainlayer import embeddings

    require(embeddings.DEFAULT_MODEL == "BAAI/bge-large-en-v1.5", "Local bge routing changed")
    from brainlayer.pipeline.longitudinal_analyzer import _ollama_generate, loopback_ollama_url

    previous_host = os.environ["OLLAMA_HOST"]
    try:
        for remote in (
            "https://fixture.invalid",
            "http://192.0.2.1:11434",
            "http://127.0.0.1.fixture.invalid",
            "http://user:pass@127.0.0.1",
            "http://127.0.0.1/api/generate",
        ):
            os.environ["OLLAMA_HOST"] = remote
            try:
                _ollama_generate(prompt="synthetic local-only adapter probe")
            except ValueError:
                pass
            else:
                raise RuntimeError("Local adapter accepted a remote/ambiguous endpoint")
        os.environ["OLLAMA_HOST"] = "http://localhost:11434"
        require(loopback_ollama_url() == "http://127.0.0.1:11434", "Local adapter delegated localhost to DNS")
    finally:
        os.environ["OLLAMA_HOST"] = previous_host
    from brainlayer.pipeline.digest import digest_content
    from brainlayer.vector_store import VectorStore

    store = VectorStore(Path(os.environ["HOME"]) / "digest.db")
    try:
        result = digest_content(
            content="Person Alpha chose BrainLayer. TODO: write local tests. Why keep local embeddings?",
            store=store,
            embed_fn=lambda text: [1.0] + [0.0] * 1023,
            participants=["Person Alpha"],
            title="Synthetic decision",
        )
        require(
            result["enrichment"] == {"status": "retired", "reason": "cloud_enrichment_retired"},
            "Digest selected a producer",
        )
        require(store.get_chunk(result["digest_id"])["content"].startswith("Person Alpha"), "Local persistence failed")
        require(
            store.conn.execute("SELECT 1 FROM chunk_vectors WHERE chunk_id=?", (result["digest_id"],)).fetchone(),
            "Fixture vector absent",
        )
    finally:
        store.close()
    # Only the identity-checked fixture directory is added, never src/ or an
    # editable package. Every brainlayer origin remains in the private wheel.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from retirement_surfaces import installed_surfaces

    surfaces = installed_surfaces(Path(os.environ["HOME"]))
    sdk = inspect_sdk(config["require_sdk_absence"])
    events = [json.loads(line) for line in Path(os.environ["RETIREMENT_EVENTS"]).read_text().splitlines()]
    require(
        len([event for event in events if event["phase"] == "control"]) == positive_count, "Control ledger incomplete"
    )
    require(
        not [event for event in events if event["phase"] == "candidate"], "Candidate attempted a forbidden operation"
    )
    require(
        not [
            name
            for name in sys.modules
            if name.split(".")[0] in {"openai", "groq", "anthropic"} or name.startswith("google.genai")
        ],
        "Model SDK loaded",
    )
    return {
        "status": "PASS",
        "origin": str(origin),
        "prefix": sys.prefix,
        "verified_members": verified,
        "modules": sorted(set(modules)),
        "controls": positive_count,
        "events": events,
        "sdk": sdk,
        "surfaces": surfaces,
        "core": "actual installed imports, Sanitizer/prompt/parser, local bge routing; actual temporary digest persistence with synthetic vectors",
        "local_adapter": "remote/credential/path overrides rejected before any send; localhost resolved numerically",
    }


if __name__ == "__main__":
    try:
        report = run(json.loads(Path(sys.argv[1]).read_text()))
    except Exception as error:
        report = {"status": "FAIL", "error": str(error)}
    Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + "\n")
    sys.exit(0 if report["status"] == "PASS" else 1)
