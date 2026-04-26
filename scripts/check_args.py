import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config.py"


def collect_declared_args(config_path: Path):
    mod = ast.parse(config_path.read_text())
    declared = set()
    for node in ast.walk(mod):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            dest = None
            for kw in node.keywords:
                if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                    dest = kw.value.value
            if dest is None:
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--"):
                        dest = arg.value[2:].replace("-", "_")
                        break
            if dest:
                declared.add(dest)
    return declared


def collect_used_args(root: Path):
    used_by_file = {}
    for path in root.rglob("*.py"):
        rel_parts = path.relative_to(root).parts
        if "__pycache__" in rel_parts or path.name == "check_args.py":
            continue
        if rel_parts and rel_parts[0] in {"legacy", "archive"}:
            continue
        if rel_parts and rel_parts[0] == "scripts":
            continue
        try:
            mod = ast.parse(path.read_text())
        except Exception:
            continue
        names = set()
        for node in ast.walk(mod):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "args":
                names.add(node.attr)
        if names:
            used_by_file[path] = names
    return used_by_file


if __name__ == "__main__":
    declared = collect_declared_args(CONFIG)
    used_by_file = collect_used_args(ROOT)
    used = set().union(*used_by_file.values()) if used_by_file else set()

    missing = sorted(used - declared)

    print(f"declared={len(declared)} used={len(used)} missing={len(missing)}")
    if missing:
        print("Missing arg definitions:", missing)
        for path, names in sorted(used_by_file.items()):
            bad = sorted(names - declared)
            if bad:
                print(path.relative_to(ROOT), bad)
        raise SystemExit(1)

    print("Argument check passed.")
