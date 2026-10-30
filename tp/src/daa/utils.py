"""Small helpers shared by the run writer and reporting code."""

from __future__ import annotations

import re
import subprocess

from .config import PROJECT_ROOT

# Only code that can change a score counts as dirty. ``outputs/`` is excluded
# on purpose: the run writer itself appends to the tracked leaderboard, so an
# unfiltered check would mark every post-first run dirty. Ignored paths
# (``data/``, ``outputs/runs/``) never appear in ``git status``.
SOURCE_DIRECTORIES = ("src", "notebooks", "splits", "tests")


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


def git_dirty() -> bool:
    """True when source code differs from the ``git_sha`` being recorded.

    Purpose: distinguish leaderboard rows whose code matches the recorded
    hash from provisional rows made with uncommitted or untracked edits.

    Invariants:
        Counts tracked modifications plus untracked *new* files under
        ``SOURCE_DIRECTORIES`` (a new module changes behaviour without being
        caught by a ``HEAD`` diff). Ignored files are never checked.

    Failure modes:
        Returns ``False`` outside a git checkout or when ``git`` is missing;
        "unknown" provenance is reported as clean rather than blocking a run.
    """
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--", *SOURCE_DIRECTORIES],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return bool(status.strip())


def slugify(value: str) -> str:
    """Turn a human name into a safe directory component."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "run"
