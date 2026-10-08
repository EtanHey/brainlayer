"""Static model-transport gate. Includes imports hidden in unexecuted bodies."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import re
from pathlib import Path

RETIRED = {
    "brainlayer.enrichment_controller",
    "brainlayer.pipeline.enrichment",
    "brainlayer.pipeline.groq",
    "enrichment_controller",
    "pipeline.enrichment",
    "pipeline.groq",
}
SDKS = {
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
}
SEND_NAMES = {"generate_content", "generate_content_stream", "generateContent", "generateContentStream"}
MODEL_URL = re.compile(
    r"https?://(?:api\.(?:openai\.com|anthropic\.com|groq\.com|cohere\.(?:ai|com)|"
    r"mistral\.ai|x\.ai|deepseek\.com|together\.(?:xyz|ai)|fireworks\.ai|perplexity\.ai)|"
    r"openrouter\.ai|generativelanguage\.googleapis\.com|"
    r"(?:[^/]+-)?aiplatform\.googleapis\.com|bedrock-runtime\.[a-z0-9-]+\.amazonaws\.com(?:\.cn)?|"
    r"[a-z0-9.-]+\.openai\.azure\.com|(?:api-inference|router)\.huggingface\.co|"
    r"[a-z0-9.-]+\.endpoints\.huggingface\.cloud)(?::\d+)?(?=[/?#\s'\"]|$)",
    re.I,
)
ROOTS = ("src", "scripts", "hooks", "brain-bar/Sources", "dashboard")
EXTENSIONS = {".py", ".swift", ".sh", ".js", ".ts", ".tsx", ".mjs", ".c", ".h", ".cpp", ".m", ".mm"}


def forbidden(name: str) -> bool:
    return any(name == item or name.startswith(item + ".") for item in RETIRED | SDKS)


def python_findings(text: str, module: str) -> list[dict]:
    tree = ast.parse(text)
    package = module.removesuffix(".__init__") if module.endswith(".__init__") else module.rpartition(".")[0]
    findings = []
    aliases: dict[str, set[str]] = {}

    def record(node, target):
        if forbidden(target):
            findings.append({"line": node.lineno, "target": target})

    def qualify(node):
        if isinstance(node, ast.Name):
            return aliases.get(node.id, {node.id})
        if isinstance(node, ast.Attribute):
            return {parent + "." + node.attr for parent in qualify(node.value)}
        return set()

    # Conservative alias union cannot erase a forbidden reference by shadowing
    # the same identifier in another function. Actual import identity is exact.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                record(node, item.name)
                aliases.setdefault(item.asname or item.name.split(".")[0], set()).add(
                    item.name if item.asname else item.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            target = node.module or ""
            if node.level:
                target = importlib.util.resolve_name("." * node.level + target, package)
            record(node, target)
            for item in node.names:
                name = target + "." + item.name
                record(node, name)
                aliases.setdefault(item.asname or item.name, set()).add(name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            for target in qualify(node):
                record(node, target)
        if isinstance(node, ast.Call):
            for target in qualify(node.func):
                record(node, target)
                if target.rsplit(".", 1)[-1] in SEND_NAMES or target.endswith(
                    (".chat.completions.create", ".messages.create")
                ):
                    findings.append({"line": node.lineno, "target": target})
                if target in {"importlib.import_module", "__import__", "builtins.__import__"} and node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        name = arg.value
                        if name.startswith("."):
                            context = next(
                                (k.value for k in node.keywords if k.arg == "package"),
                                node.args[1] if len(node.args) > 1 else None,
                            )
                            if not isinstance(context, ast.Constant) or not isinstance(context.value, str):
                                raise ValueError("Relative dynamic import has no constant package")
                            name = importlib.util.resolve_name(name, context.value)
                        record(node, name)
    unique = {(row["line"], row["target"]): row for row in findings}
    return sorted(unique.values(), key=lambda row: (row["line"], row["target"]))


def source_scan(root: Path) -> dict:
    inventory, findings, errors = {}, [], []
    for directory in ROOTS:
        for path in sorted((root / directory).rglob("*")):
            if path.suffix not in EXTENSIONS or not path.is_file():
                continue
            if set(path.relative_to(root).parts) & {"node_modules", ".next", ".build", ".swiftpm", "__pycache__"}:
                continue  # generated third-party/build trees, not project sources
            relative = path.relative_to(root).as_posix()
            try:
                if path.is_symlink() and not path.resolve().is_relative_to(root.resolve()):
                    raise ValueError("Source symlink escapes immutable tree")
                data = path.read_bytes()
                inventory[relative] = hashlib.sha256(data).hexdigest()
                text = data.decode()
                if path.suffix == ".py":
                    module = relative.removeprefix("src/").removesuffix(".py").replace("/", ".")
                    findings.extend({"path": relative, **row} for row in python_findings(text, module))
                for number, line in enumerate(text.splitlines(), 1):
                    if MODEL_URL.search(line):
                        findings.append({"path": relative, "line": number, "target": "cloud model URL"})
                    if path.suffix != ".py" and re.search(
                        r"\b(?:generateContent(?:Stream)?|generate_content(?:_stream)?)\s*\(|"
                        r"\bimport\s+(?:OpenAI|GoogleGenerativeAI|Anthropic)|"
                        r"(?:from|require\s*\()\s*['\"](?:openai|@google/(?:genai|generative-ai)|@anthropic-ai/sdk)['\"]",
                        line,
                    ):
                        findings.append({"path": relative, "line": number, "target": "model transport syntax"})
            except (OSError, UnicodeError, SyntaxError, ValueError, ImportError) as error:
                errors.append({"path": relative, "error": str(error)})
    if not any(name.startswith("src/brainlayer/") for name in inventory):
        errors.append({"path": "src/brainlayer", "error": "Package inventory empty"})
    if not any(name.startswith("brain-bar/Sources/") for name in inventory):
        errors.append({"path": "brain-bar/Sources", "error": "Native source inventory empty"})
    digest = hashlib.sha256(
        "\n".join(f"{name}:{value}" for name, value in sorted(inventory.items())).encode()
    ).hexdigest()
    return {"inventory": inventory, "inventory_sha256": digest, "findings": findings, "errors": errors}
