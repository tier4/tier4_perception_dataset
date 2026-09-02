#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path
import shlex
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: parse_runtime_build_yaml.py CONFIG.yaml", file=sys.stderr)
        return 2
    values = parse_yaml_subset(Path(sys.argv[1]))
    for key, value in values.items():
        if os.environ.get(key):
            continue
        print(f"{key}={shlex.quote(value)}")
    return 0


def parse_yaml_subset(path: Path) -> dict[str, str]:
    root: dict[str, object] = {}
    stack: list[tuple[int, dict[str, object]]] = [(-1, root)]

    for lineno, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "\t" in raw_line:
            raise ValueError(f"{path}:{lineno}: tabs are not supported")
        if ":" not in stripped:
            raise ValueError(f"{path}:{lineno}: expected key: value")

        indent = len(line) - len(line.lstrip(" "))
        key, raw_value = stripped.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            raise ValueError(f"{path}:{lineno}: empty key")

        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"{path}:{lineno}: invalid indentation")
        parent = stack[-1][1]

        if raw_value == "":
            child: dict[str, object] = {}
            parent[key] = child
            stack.append((indent, child))
            continue

        parent[key] = _unquote(raw_value)

    return flatten_config(root)


def flatten_config(root: dict[str, object]) -> dict[str, str]:
    values: dict[str, str] = {}
    for key in ("ros_distro", "image_name", "image_tag"):
        value = root.get(key)
        if isinstance(value, str):
            values[key.upper()] = value

    repos = root.get("repos")
    if isinstance(repos, dict):
        for name, repo_config in repos.items():
            if not isinstance(name, str) or not isinstance(repo_config, dict):
                continue
            prefix = name.upper()
            repo = repo_config.get("repo")
            ref = repo_config.get("ref")
            if isinstance(repo, str):
                values[f"{prefix}_REPO"] = repo
            if isinstance(ref, str):
                values[f"{prefix}_REF"] = ref
    return values


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
