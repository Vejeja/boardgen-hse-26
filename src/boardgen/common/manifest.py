from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from boardgen.common.utils import ensure_dir, file_sha256


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str | None:
    if not path or not os.path.exists(path):
        return None
    return file_sha256(path, chunk_size)


def git_state(repo: str = ".") -> dict[str, Any]:
    def run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args],
                cwd=repo,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return None

    status = run(["status", "--short"])
    return {
        "sha": run(["rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_short": status,
    }


def package_versions(names: list[str] | None = None) -> dict[str, str | None]:
    names = names or [
        "boardgen",
        "numpy",
        "opencv-python",
        "Pillow",
        "PyYAML",
        "scikit-image",
        "trimesh",
        "scipy",
        "torch",
    ]
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def build_manifest(
    *,
    config_paths: list[str] | None = None,
    source_videos: list[str] | None = None,
    components: dict[str, Any] | None = None,
    render: dict[str, Any] | None = None,
    random_seed: int | None = None,
    environment_id: str | None = None,
) -> dict[str, Any]:
    config_paths = config_paths or []
    source_videos = source_videos or []
    return {
        "schema_version": "boardgen.run_manifest.v1",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git": git_state(),
        "config_hashes": {p: sha256_file(p) for p in config_paths},
        "source_video_hashes": {p: sha256_file(p) for p in source_videos},
        "components": components or {},
        "render": render or {},
        "random_seed": random_seed,
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "environment_id": environment_id or os.getenv("BOARDGEN_ENV_ID"),
    }


def write_manifest(path: str, manifest: dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def source_provenance() -> dict:
    package = Path(__file__).resolve().parents[1]
    return {
        "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True)),
        "platform": platform.platform(),
        "seed": 42,
        "source_hashes": {
            str(p.relative_to(package)): file_sha256(p) for p in sorted(package.rglob("*.py"))
        },
        "versions": {
            name: metadata.version(name) for name in ("numpy", "opencv-python", "trimesh", "scipy")
        },
    }
