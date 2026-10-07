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

    def resolve(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return resolve(node.value) + "." + node.attr
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            return resolve(node.args[0]) + "." + node.args[1].value
        if isinstance(node, ast.Call):
            callee = resolve(node.func)
            if transport_import(callee) and callee.rsplit(".", 1)[-1] in factories:
                return callee + "()"
            return "<call-result>"
        target = ast.unparse(node)
        first, dot, remainder = target.partition(".")
        return aliases.get(first, first) + (dot + remainder if dot else "")

    def callable_target(target):
        last = target.rsplit(".", 1)[-1]
        return target not in {"re.compile", "platform.system"} and (
            last in NETWORK_NAMES | DYNAMIC_NAMES | PROCESS_NAMES | {"get"}
            or (transport_import(target) and not last.isupper())
        )

    # Bind callable values, not arbitrary return values (e.g. CompletedProcess data).
    factories = {"Session", "Client", "AsyncClient", "ClientSession", "socket", "SSLContext", "create_connection"}
    bindings = [node for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr))]
    for _ in range(len(bindings) + 1):
        changed = False
        for node in bindings:
            value = node.value
            if value is None or not isinstance(value, (ast.Name, ast.Attribute, ast.Call)):
                continue
            if isinstance(value, ast.Name) and value.id not in aliases and value.id not in DYNAMIC_NAMES:
                continue
            if isinstance(value, ast.Call):
                callee = resolve(value.func)
                if transport_import(callee) and callee.rsplit(".", 1)[-1] in factories:
                    target = callee + "()"
                elif isinstance(value.func, ast.Name) and value.func.id == "getattr":
                    target = resolve(value)
                else:
                    continue
            else:
                target = resolve(value)
            if not callable_target(target):
                continue
            names = node.targets if isinstance(node, ast.Assign) else [node.target]
            for name in names:
                if isinstance(name, ast.Name) and name.id not in aliases:
                    aliases[name.id] = target
                    changed = True
        if not changed:
            break
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}

    def walk(node, owner):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            owner = owner + "." + node.name
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Call)):
            target = ast.unparse(node.func) if isinstance(node, ast.Call) else ast.unparse(node)
            if isinstance(node, ast.Call):
                target = resolve(node.func)
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
            if (
                isinstance(node, ast.Call)
                and kind is None
                and ast.unparse(node.func).split(".")[0] in aliases
                and transport_import(target)
            ):
                kind = "transport_call"
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and callable_target(resolve(node))
            ):
                kind = "callable_reference"
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
        if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
            parent = parents.get(node)
            # Direct calls are already bound above; classify only the outer escaping expression.
            direct = isinstance(parent, ast.Call) and parent.func is node
            nested = isinstance(parent, ast.Attribute) and parent.value is node
            target = resolve(node)
            known = not isinstance(node, ast.Name) or node.id in aliases or node.id in DYNAMIC_NAMES
            if known and not direct and not nested and callable_target(target):
                sites.append(
                    {
                        "path": path,
                        "owner": owner,
                        "kind": "callable_reference",
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
