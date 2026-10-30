"""Small helpers shared by the run writer and reporting code."""

from __future__ import annotations

import re
import subprocess

from .config import PROJECT_ROOT


def git_sha() -> str:
    """Short HEAD hash, or ``"unknown"`` outside a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def slugify(value: str) -> str:
    """Turn a human name into a safe directory component."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "run"
