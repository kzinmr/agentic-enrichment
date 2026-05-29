from __future__ import annotations

import os
from pathlib import Path
import re


ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def default_env_file() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [cwd / ".env", *(parent / ".env" for parent in cwd.parents)]
    package_path = Path(__file__).resolve()
    candidates.extend(parent / ".env" for parent in package_path.parents)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return cwd / ".env"


def load_env_file(path: str | Path | None, *, override: bool = False) -> list[str]:
    if path is None:
        return []
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return []

    loaded_keys: list[str] = []
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid .env line {line_number} in {env_path}: expected KEY=VALUE")

        key, value = line.split("=", 1)
        key = key.strip()
        if not ENV_KEY_RE.fullmatch(key):
            raise ValueError(f"Invalid .env key on line {line_number} in {env_path}: {key!r}")
        if key in os.environ and not override:
            continue

        os.environ[key] = parse_env_value(value.strip())
        loaded_keys.append(key)
    return loaded_keys


def parse_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        quote = value[0]
        value = value[1:-1]
        if quote == '"':
            return value.replace("\\n", "\n").replace('\\"', '"').replace("\\\\", "\\")
        return value
    if " #" in value:
        value = value.split(" #", 1)[0].rstrip()
    return value
