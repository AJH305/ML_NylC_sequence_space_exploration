"""Content hashes, software identity and durable JSON provenance."""

from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import distributions
from pathlib import Path
import json
import math
import platform
import subprocess
import sys
import numpy as np


def sha256_file(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(
        json.dumps(clean_json(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def digest_json(value):
    return sha256(
        json.dumps(clean_json(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def software_identity():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": dict(
            sorted(
                (d.metadata["Name"].lower(), d.version)
                for d in distributions()
                if d.metadata["Name"]
            )
        ),
    }


def code_identity(root):
    root = Path(root)
    files = (
        sorted((root / "src").rglob("*.py"))
        + sorted((root / "src").rglob("*.json"))
        + sorted((root / "workflow").rglob("*.smk"))
    )
    files += [
        p
        for p in [
            root / "workflow/Snakefile",
            root / "uv.lock",
            root / "pyproject.toml",
            root / ".python-version",
        ]
        if p.exists()
    ]
    hashes = {p.relative_to(root).as_posix(): sha256_file(p) for p in files}

    def git(*args):
        try:
            return subprocess.run(
                ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "sha256": digest_json(hashes),
        "files": hashes,
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain", "--", "."),
    }


def utc_now():
    return datetime.now(timezone.utc).isoformat()
