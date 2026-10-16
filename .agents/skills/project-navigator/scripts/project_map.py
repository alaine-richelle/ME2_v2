"""Print a compact, read-only navigation map for the ME2_v2 repository."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

IGNORED_PARTS = {
    ".git",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "data",
    "logs",
    "models",
    "results",
}
SOURCE_SUFFIXES = {".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
SYMBOL_PATTERN = re.compile(r"^(?:async\s+)?(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE)


def repository_root() -> Path:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        return Path(result.stdout.strip()).resolve()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return Path(__file__).resolve().parents[4]


def source_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if any(part in IGNORED_PARTS for part in relative.parts):
            continue
        if path.is_file() and path.suffix in SOURCE_SUFFIXES:
            files.append(relative)
    return sorted(files)


def symbols_for(root: Path, relative: Path) -> list[str]:
    if relative.suffix != ".py":
        return []
    try:
        text = (root / relative).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    return SYMBOL_PATTERN.findall(text)


def git_status(root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ["(Git status unavailable)"]
    return result.stdout.splitlines() or ["(clean)"]


def main() -> None:
    root = repository_root()
    print(f"# Project map: {root.name}")
    print(f"\nRoot: `{root}`")

    print("\n## Navigable source and control files")
    for relative in source_files(root):
        symbols = symbols_for(root, relative)
        detail = f" — {', '.join(symbols)}" if symbols else ""
        print(f"- `{relative}`{detail}")

    print("\n## Artifact roots")
    for name in ("wake-word/data", "wake-word/models", "vcm/data", "vcm/models", "results", "logs"):
        path = root / name
        state = "present" if path.exists() else "missing"
        print(f"- `{name}/`: {state}")

    print("\n## Git status")
    for line in git_status(root):
        print(f"- `{line}`")


if __name__ == "__main__":
    main()
