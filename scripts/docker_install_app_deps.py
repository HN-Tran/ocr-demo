#!/usr/bin/env python3
"""Install docread without replacing torch/torchvision from the base image."""
from __future__ import annotations

import os
import subprocess
import sys
import tomllib
from pathlib import Path

APP_ROOT = Path(os.environ.get("APP_ROOT", "/app"))
SKIP = frozenset({"torch", "torchvision"})


def _package_name(spec: str) -> str:
    return spec.split("[")[0].strip().split("<")[0].split(">")[0].split("=")[0].strip().lower()


def _filter_specs(specs: list[str]) -> list[str]:
    return [s for s in specs if _package_name(s) not in SKIP]


def main() -> None:
    data = tomllib.loads((APP_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = _filter_specs(list(data["project"]["dependencies"]))
    extra = os.environ.get("INSTALL_EXTRA", "").strip()
    if extra:
        deps.extend(_filter_specs(list(data["project"]["optional-dependencies"].get(extra, []))))

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--upgrade", "pip", "setuptools", "wheel"]
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps", str(APP_ROOT)]
    )
    if deps:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", *deps])


if __name__ == "__main__":
    main()
