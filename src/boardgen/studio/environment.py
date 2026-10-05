from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .storage import sha

PACKAGE = Path(__file__).resolve().parents[1]
CHECKOUT = PACKAGE.parents[1] if (PACKAGE.parents[1] / "pyproject.toml").exists() else None
REPOSITORY = (
    Path(os.environ.get("BOARDGEN_HOME", str(CHECKOUT or Path.home() / ".cache/boardgen")))
    .expanduser()
    .resolve()
)
TOOLS = (
    Path(os.environ.get("BOARDGEN_TOOLS_DIR", str(REPOSITORY / "tmp/benchmark_tools")))
    .expanduser()
    .resolve()
)




def spirula():
    return Path(
        os.environ.get("SPIRULA_BIN", str(TOOLS / "Spirula Studio.app/Contents/MacOS/spirula"))
    )


def package(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check(project):
    recipes = [project.prepare.resolved(f.id) for f in project.figures] or [
        project.prepare.defaults
    ]
    paths = {"FFmpeg": shutil.which("ffmpeg")}
    if any(r.shape.method == "mv" for r in recipes):
        paths.update(
            {
                "MV Turbo: веса": TOOLS / "hunyuan-mv-turbo/model.fp16.safetensors",
                "MV Turbo: MLX-код": TOOLS / "hunyuan3d-swift/python/shape/hy3dmlx/pipeline.py",
            }
        )
    if any(r.shape.method == "spirula" or r.appearance.registration == "orbit" for r in recipes):
        paths["Spirula"] = spirula()
    if any(r.appearance.method == "photo" for r in recipes):
        paths.update(
            {
                "OpenMVS: текстуры": TOOLS / "openmvs/TextureMesh",
                "OpenMVS: импорт": TOOLS / "openmvs/InterfaceCOLMAP",
            }
        )
    names = ["pymeshlab"]
    if any(r.matte.method == "birefnet" for r in recipes):
        names.extend(["torch", "transformers"])
    if any(r.shape.method == "mv" for r in recipes):
        names.append("mlx")
    items = [
        {
            "name": name,
            "ready": bool(path and Path(path).exists()),
            "path": str(path or ""),
            "type": "tool",
        }
        for name, path in paths.items()
    ]
    items.extend(
        {"name": n, "ready": package(n) is not None, "version": package(n), "type": "package"}
        for n in names
    )
    return {
        "ready": all(i["ready"] for i in items),
        "items": items,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
    }


def fingerprint():
    code = {str(p.relative_to(PACKAGE)): sha(p) for p in sorted(PACKAGE.rglob("*.py"))}
    external_code = {
        str(p.relative_to(TOOLS)): sha(p)
        for p in sorted((TOOLS / "hunyuan3d-swift/python/shape/hy3dmlx").rglob("*.py"))
    }
    files = [
        TOOLS / "hunyuan-mv-turbo/model.fp16.safetensors",
        TOOLS / "hunyuan-mv-turbo/config.yaml",
        spirula(),
        TOOLS / "openmvs/TextureMesh",
    ]
    neural = {}
    try:
        from huggingface_hub import try_to_load_from_cache

        from boardgen.video.neuralseg import DEFAULT_MODEL, FALLBACK_MODEL

        for model in {DEFAULT_MODEL, FALLBACK_MODEL}:
            config = try_to_load_from_cache(model, "config.json")
            if isinstance(config, str):
                folder = Path(config).parent
                neural[model] = {
                    "revision": folder.name,
                    "files": {p.name: sha(p) for p in folder.iterdir() if p.is_file()},
                }
    except ImportError:
        pass
    return {
        "neural": neural,
        "code": code,
        "external_code": external_code,
        "tools": {str(p): sha(p) for p in files if p.is_file()},
        "packages": {
            n: package(n)
            for n in [
                "torch",
                "transformers",
                "mlx",
                "pymeshlab",
                "trimesh",
                "numpy",
                "opencv-python",
                "scipy",
                "Pillow",
            ]
        },
    }


