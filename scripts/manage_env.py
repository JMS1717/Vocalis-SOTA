#!/usr/bin/env python3
"""Utilities for reading and writing simple .env files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple


def _load_env_lines(path: Path) -> List[Tuple[str, str | None]]:
    lines: List[Tuple[str, str | None]] = []
    if not path.exists():
        return lines

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle.readlines():
            line = raw_line.rstrip("\n")
            if not line or line.lstrip().startswith("#"):
                lines.append((line, None))
                continue

            if "=" not in line:
                lines.append((line, None))
                continue

            key, value = line.split("=", 1)
            lines.append((key, value))

    return lines


def _write_env_lines(path: Path, lines: List[Tuple[str, str | None]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for entry, value in lines:
            if value is None:
                handle.write(f"{entry}\n")
            else:
                handle.write(f"{entry}={value}\n")


def get_value(path: Path, key: str) -> str | None:
    for entry, value in _load_env_lines(path):
        if value is not None and entry == key:
            return value
    return None


def set_value(path: Path, key: str, value: str) -> None:
    lines = _load_env_lines(path)
    updated: List[Tuple[str, str | None]] = []
    replaced = False

    for entry, current_value in lines:
        if current_value is not None and entry == key:
            updated.append((entry, value))
            replaced = True
        else:
            updated.append((entry, current_value))

    if not replaced:
        updated.append((key, value))

    _write_env_lines(path, updated)


def remove_value(path: Path, key: str) -> None:
    lines = _load_env_lines(path)
    updated = [entry for entry in lines if not (entry[1] is not None and entry[0] == key)]

    if updated:
        _write_env_lines(path, updated)
    elif path.exists():
        path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect or update a .env file")
    parser.add_argument("--path", default=".env", help="Path to the .env file (default: %(default)s)")

    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--get", metavar="KEY", help="Print the value for KEY if it exists")
    action.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"), help="Set KEY to VALUE, creating the file if necessary")
    action.add_argument("--unset", metavar="KEY", help="Remove KEY from the file if it exists")

    args = parser.parse_args()
    env_path = Path(args.path).expanduser().resolve()

    if args.get:
        value = get_value(env_path, args.get)
        if value is not None:
            print(value)
        return 0

    if args.set:
        key, value = args.set
        env_path.parent.mkdir(parents=True, exist_ok=True)
        set_value(env_path, key, value)
        return 0

    if args.unset:
        remove_value(env_path, args.unset)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
