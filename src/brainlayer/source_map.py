"""Manual, isolated cmuxlayer maps. No memory DB, hooks, or resident worker."""

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

VERSION = "0.9.96"
ENGINE_SHA = "d74d6493b85f23b0fa482d5d9ddb52cfc13864ce0dd3fc511ca265094aa1f17c"
FALLBACK = "Use rg directly; enqueue source-map refresh, then explicitly run refresh --run."


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True).encode()


def graph_nodes(path):
    data = json.loads(path.read_text())
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("nodes"), list)
        or not isinstance(data.get("links", data.get("edges")), list)
    ):
        raise ValueError("invalid graph shape")
    nodes = data["nodes"]
    if any(
        not isinstance(node, dict)
        or not isinstance(node.get("id"), str)
        or not node["id"]
        or not isinstance(node.get("label", ""), str)
        for node in nodes
    ):
        raise ValueError("invalid graph node")
    return nodes


def atomic(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
        f.write(encoded(value))
        f.flush()
        os.fsync(f.fileno())
    os.replace(f.name, path)


def git(repo, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.check_output(["git", "-C", str(repo), *args], env=env, stderr=subprocess.PIPE, timeout=10)


def engine_identity(python):
    script = """import hashlib,json,re
from importlib.metadata import distribution,distributions
p=distribution('graphifyy'); root=p.locate_file('graphify'); h=hashlib.sha256()
for f in sorted(root.rglob('*.py')):
 h.update(str(f.relative_to(root)).encode()+b'\\0'+f.read_bytes()+b'\\0')
g=sorted((d.metadata['Name'],d.version) for d in distributions() if d.metadata['Name'].lower().replace('_','-').startswith('tree-sitter'))
print(json.dumps(dict(version=p.version,sha=h.hexdigest(),epoch=int(re.search(r'_AST_CACHE_SCHEMA = (\\d+)',(root/'cache.py').read_text())[1]),grammars=g)))"""
    identity = json.loads(
        subprocess.check_output([str(python), "-I", "-B", "-c", script], stderr=subprocess.PIPE, timeout=15)
    )
    if (identity["version"], identity["sha"], identity["epoch"]) != (VERSION, ENGINE_SHA, 27):
        raise ValueError("incompatible engine: require unmodified Graphify 0.9.96, extractor epoch 27")
    return identity


def run_graphify(python, args, cwd, out):
    home = out / "home"
    home.mkdir(exist_ok=True)
    env = dict(
        PATH=os.defpath,
        HOME=str(home),
        GRAPHIFY_OUT=str(out),
        GRAPHIFY_BACKEND="none",
        GRAPHIFY_MAX_WORKERS="1",
        GRAPHIFY_VIZ_NODE_LIMIT="0",
        PYTHONHASHSEED="0",
        GRAPHIFY_NO_AUTO_REFRESH="1",
    )
    with subprocess.Popen(
        [str(python), "-I", "-B", "-m", "graphify", *args],
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
    (out / f"{args[0]}.stdout").write_text(stdout)
    (out / f"{args[0]}.stderr").write_text(stderr)
    if process.returncode:
        raise RuntimeError(f"Graphify {args[0]} exited {process.returncode}; see {out}")
    return stdout


class SourceMap:
    def __init__(self, repo, state_root, python):
        self.repo = Path(repo).resolve()
        if Path(git(self.repo, "rev-parse", "--show-toplevel").decode().strip()).resolve() != self.repo:
            raise ValueError("repo must be the worktree root")
        remote = git(self.repo, "remote", "get-url", "origin").decode().strip()
        if remote not in (
            "https://github.com/EtanHey/cmuxlayer.git",
            "https://github.com/EtanHey/cmuxlayer",
            "git@github.com:EtanHey/cmuxlayer.git",
        ):
            raise ValueError("repo is outside the cmuxlayer allowlist")
        root = Path(state_root).resolve()
        if root == self.repo or self.repo in root.parents:
            raise ValueError("state-root must be outside the source worktree")
        self.state = root / digest(str(self.repo).encode())[:24]
        self.python = Path(python).absolute()

    def fingerprint(self):
        head = git(self.repo, "rev-parse", "HEAD").decode().strip()
        names = sorted(
            set(git(self.repo, "ls-files", "-z", "--cached", "--others", "--exclude-standard").decode().split("\0"))
            - {""}
        )
        for directory in {parent for name in names for parent in Path(name).parents}:
            for basename in (".gitignore", ".graphifyignore"):
                name = str(directory / basename)
                if (self.repo / name).is_file() and name not in names:
                    names.append(name)
        files = {}
        for name in names:
            path = self.repo / name
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise ValueError(f"unsupported symlink/submodule input: {name}")
            files[name] = digest(path.read_bytes()) if path.exists() else None
        config = git(self.repo, "config", "--null", "--list", "--show-origin")
        for path in (
            Path(git(self.repo, "rev-parse", "--git-path", "info/exclude").decode().strip()),
            Path(
                git(self.repo, "config", "--default", "", "--path", "--get", "core.excludesFile").decode().strip()
                or "/dev/null"
            ),
        ):
            path = path if path.is_absolute() else self.repo / path
            config += str(path).encode() + (path.read_bytes() if path.exists() else b"")
        if head != git(self.repo, "rev-parse", "HEAD").decode().strip():
            raise ValueError("HEAD changed during fingerprint")
        dirty = git(self.repo, "status", "--porcelain=v1", "-z", "--untracked-files=all")
        ignores = {k: v for k, v in files.items() if Path(k).name in (".gitignore", ".graphifyignore")}
        return dict(
            repo=str(self.repo),
            head=head,
            files=files,
            dirty_digest=digest(dirty + encoded(files)),
            config_digest=digest(config + encoded(ignores)),
        )

    def doctor(self):
        try:
            current = json.loads((self.state / "current.json").read_text())
            if not isinstance(current, dict):
                raise ValueError("invalid receipt")
            if current.get("status") == "failed":
                return dict(current, reason=f"{current['reason']}; {FALLBACK}")
            graph = Path(current["graph"])
            if not graph.is_file():
                return dict(current, status="missing", reason=FALLBACK)
            if digest(graph.read_bytes()) != current["checksum"]:
                return dict(current, status="corrupt", reason=FALLBACK)
            if current["status"] != "fresh":
                return dict(current, reason=FALLBACK)
            if current["engine"] != engine_identity(self.python):
                return dict(current, status="incompatible", reason=FALLBACK)
            if current["inputs"] != self.fingerprint():
                return dict(current, status="stale", reason=FALLBACK)
            return current
        except FileNotFoundError:
            return dict(status="missing", reason=FALLBACK)
        except (ValueError, OSError, subprocess.SubprocessError, KeyError, TypeError) as exc:
            return dict(status="refused", reason=f"{exc}; {FALLBACK}")

    def refresh(self, run=False):
        self.state.mkdir(parents=True, exist_ok=True)
        if not run:
            request = dict(request=uuid.uuid4().hex, inputs=self.fingerprint(), requested_at=time.time())
            atomic(self.state / "queue.json", request)
            return dict(request, status="queued", next="Explicitly execute refresh --run in owned scratch.")
        with (self.state / "worker.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return dict(status="busy", reason="An owned refresh is running; latest request remains queued.")
            try:
                prior = json.loads((self.state / "current.json").read_text())
            except (OSError, ValueError):
                prior = {}
            if not isinstance(prior, dict):
                prior = {}
            request_id = None
            try:
                try:
                    request = json.loads((self.state / "queue.json").read_text())
                except FileNotFoundError:
                    return self.doctor()
                request_id = request["request"]
                if request_id == prior.get("consumed_request", prior.get("request")):
                    return self.doctor()
                inputs = self.fingerprint()
                engine = engine_identity(self.python)
                generation = self.state / request["request"]
                generation.mkdir(exist_ok=False)
                source, out = generation / "cmuxlayer", generation / "out"
                source.mkdir()
                out.mkdir()
                for name, checksum in inputs["files"].items():
                    if checksum is None:
                        continue
                    content = (self.repo / name).read_bytes()
                    if digest(content) != checksum:
                        raise ValueError("input changed while copying")
                    target = source / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                git(source, "init", "-q")  # Boundary: never inherit parent ignore rules.
                git(source, "add", "--force", "--all")
                run_graphify(self.python, ["update", str(source)], source, out)
                graph = out / "graph.json"
                graph_nodes(graph)
                if inputs != self.fingerprint() or engine != engine_identity(self.python):
                    raise ValueError("inputs or engine changed during build; refresh again")
                current = dict(
                    status="fresh",
                    graph=str(graph),
                    checksum=digest(graph.read_bytes()),
                    inputs=inputs,
                    engine=engine,
                    request=request["request"],
                    built_at=time.time(),
                )
                atomic(self.state / "current.json", current)
                return current
            except (ValueError, OSError, subprocess.SubprocessError, KeyError, TypeError, RuntimeError) as exc:
                failed = dict(
                    prior, status="failed", reason=str(exc), failed_at=time.time(), consumed_request=request_id
                )
                atomic(self.state / "current.json", failed)
                return failed

    def query(self, symbol, mode="affected"):
        if not symbol or mode not in ("affected", "explain"):
            return dict(
                status="refused", reason=f"Explicit exact --symbol and affected/explain mode required. {FALLBACK}"
            )
        current = self.doctor()
        if current["status"] != "fresh":
            return current
        graph = Path(current["graph"])
        try:
            matches = [
                n
                for n in graph_nodes(graph)
                if symbol in (n["id"], n.get("label"), n.get("label", "").removesuffix("()"))
            ]
            if len(matches) != 1:
                return dict(
                    status="refused",
                    reason=f"Exact symbol has {len(matches)} matches; use a unique node ID. {FALLBACK}",
                )
            output = run_graphify(
                self.python, [mode, matches[0]["id"], "--graph", str(graph)], graph.parent, graph.parent
            )
            checked = self.doctor()
            if checked != current:
                return dict(checked, status="stale", reason="Evidence changed during query; discard result.")
            return dict(current, mode=mode, symbol=symbol, pointers=output)
        except (ValueError, OSError, subprocess.SubprocessError, RuntimeError) as exc:
            return dict(current, status="refused", reason=str(exc))
