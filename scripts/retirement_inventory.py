"""Closed inventory for generic sends/dynamic code that SDK-name bans cannot cover.

The policy binds a reviewed non-cloud purpose to each transport import/call AST.
A new/changed site requires an explicit policy diff and pair review, never a skip.
"""

import ast
import hashlib
import json
import re

from scripts.retirement_source import EXTENSIONS, ROOTS, SDKS

TRANSPORT_IMPORTS = {
    "requests",
    "httpx",
    "http.client",
    "socket",
    "ssl",
    "aiohttp",
    "subprocess",
    "importlib",
    "ctypes",
    "boto3",
    "botocore",
    "sagemaker",
    "replicate",
    "deepseek",
    "fireworks",
    "perplexity",
    "huggingface_hub",
} | SDKS
NATIVE_TRANSPORT = re.compile(r"\b(?:URLSession|NWConnection|CFStream|curl)\b|\b(?:fetch|socket)\s*\(")


def transport_import(name):
    return name.startswith(("urllib", "websocket")) or any(
        name == prefix or name.startswith(prefix + ".") for prefix in TRANSPORT_IMPORTS
    )


NETWORK_NAMES = {
    "post",
    "put",
    "patch",
    "request",
    "send",
    "sendall",
    "sendto",
    "urlopen",
    "create_connection",
    "connect",
    "connect_ex",
    "handle_request",
    "handle_async_request",
}
DYNAMIC_NAMES = {"import_module", "__import__", "eval", "exec", "compile", "CDLL", "PyDLL", "dlopen"}
PROCESS_NAMES = {"Popen", "check_output", "check_call", "posix_spawn", "system", "execv", "execve", "execvp"}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def python_sites(text, path):
    tree = ast.parse(text)
    sites = []
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".")[0]] = item.name if item.asname else item.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom):
            for item in node.names:
                aliases[item.asname or item.name] = (node.module or "") + "." + item.name

    def walk(node, owner):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            owner = owner + "." + node.name
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call)):
            target = ast.unparse(node.func) if isinstance(node, ast.Call) else ast.unparse(node)
            if isinstance(node, ast.Call):
                first, dot, remainder = target.partition(".")
                target = aliases.get(first, first) + (dot + remainder if dot else "")
            last = target.rsplit(".", 1)[-1]
            kind = None
            if isinstance(node, ast.Import) and any(transport_import(item.name) for item in node.names):
                kind = "import_binding"
            elif isinstance(node, ast.ImportFrom) and any(
                transport_import((node.module or "") + "." + item.name) or transport_import(node.module or "")
                for item in node.names
            ):
                kind = "import_binding"
            if isinstance(node, ast.Call) and (
                last in NETWORK_NAMES
                or (
                    last == "get"
                    and any(word in target.lower() for word in ("client", "session", "requests", "httpx", "urllib"))
                )
            ):
                kind = "generic_send"
            elif isinstance(node, ast.Call) and last in DYNAMIC_NAMES and target != "re.compile":
                kind = "dynamic_code_or_import"
            elif (
                isinstance(node, ast.Call)
                and target != "platform.system"
                and (last in PROCESS_NAMES or (last == "run" and "subprocess" in target))
            ):
                kind = "child_process"
            if kind:
                sites.append(
                    {
                        "path": path,
                        "owner": owner,
                        "kind": kind,
                        "target": target,
                        "call_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest(),
                    }
                )
        for child in ast.iter_child_nodes(node):
            walk(child, owner)

    walk(tree, "<module>")
    return sites


def inventory(root):
    sites = {}
    for directory in ROOTS:
        for path in sorted((root / directory).rglob("*")):
            if not path.is_file() or path.suffix not in EXTENSIONS | {".c", ".h", ".cpp", ".m", ".mm"}:
                continue
            if set(path.relative_to(root).parts) & {"node_modules", ".next", ".build", ".swiftpm", "__pycache__"}:
                continue
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("Inventory source escapes snapshot")
            name = path.relative_to(root).as_posix()
            text = path.read_text()
            if path.suffix == ".py":
                rows = python_sites(text, name)
            else:
                # Bind exact bytes only for non-Python transport-bearing files.
                rows = (
                    [
                        {
                            "path": name,
                            "kind": "non_python_source",
                            "body_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        }
                    ]
                    if NATIVE_TRANSPORT.search(text)
                    else []
                )
            for row in rows:
                sites[fingerprint(row)] = row
    return dict(sorted(sites.items()))


def check_policy(root):
    actual = inventory(root)
    policy_path = root / "scripts/retirement_policy.json"
    policy = json.loads(policy_path.read_text())
    if policy.get("schema_version") != 1 or not isinstance(policy.get("sites"), dict):
        raise ValueError("Unknown/missing closed transport policy")
    unknown = []
    for key, site in actual.items():
        admitted = policy["sites"].get(key)
        if not admitted or admitted.get("site") != site or not admitted.get("non_cloud_purpose"):
            unknown.append(site)
    from scripts.retirement_source import source_scan

    return {
        "sites": actual,
        "sha256": fingerprint(actual),
        "source_inventory": source_scan(root)["inventory"],
        "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
        "unclassified": unknown,
    }
