import ast, subprocess, sys
from pathlib import Path

for path in sys.argv[1:]:
    p = Path(path)
    if not p.exists():
        continue
    tree = ast.parse(p.read_text(encoding="utf-8"))
    diff = subprocess.run(["git","diff","origin/main...HEAD","-U0","--",path],capture_output=True,text=True).stdout
    added, cur = set(), 0
    for line in diff.splitlines():
        if line.startswith("@@"):
            cur = int(line.split("+")[1].split(",")[0].split(" ")[0])
        elif line.startswith("+") and not line.startswith("+++"):
            added.add(cur); cur += 1
        elif not line.startswith("-"):
            cur += 1
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body = [b for b in node.body if not (isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) and isinstance(b.value.value, str))]
            if len(body) == 1 and isinstance(body[0], (ast.Pass, ast.Ellipsis)):
                if node.lineno in added:
                    print(f"  EMPTY-BODY (MINE) {path}:{node.lineno} {node.name}()")
                else:
                    print(f"  empty-body (pre-existing) {path}:{node.lineno} {node.name}()")
