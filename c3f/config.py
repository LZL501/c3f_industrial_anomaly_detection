from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def merge_overrides(
    config: dict[str, Any], overrides: dict[str, Any]
) -> dict[str, Any]:
    merged = copy.deepcopy(config)
    for dotted_key, value in overrides.items():
        cursor = merged
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Cannot override nested key under {dotted_key}")
        cursor[parts[-1]] = value
    return merged


def parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_dotlist(items: list[str]) -> dict[str, Any]:
    overrides = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got {item}")
        key, value = item.split("=", 1)
        overrides[key] = parse_value(value)
    return overrides
