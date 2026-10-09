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


def blender():
    explicit = os.environ.get("BLENDER_BIN")
    local = TOOLS / "blender/Blender.app/Contents/MacOS/Blender"
    return explicit or (
        str(local)
        if local.is_file()
        else shutil.which("blender") or "/Applications/Blender.app/Contents/MacOS/Blender"
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
    paths = {"FFmpeg": shutil.which("ffmpeg"), "Blender": blender()}
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
    names = ["ultralytics", "pymeshlab"]
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
        Path(blender()),
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
                "ultralytics",
                "pymeshlab",
                "trimesh",
                "numpy",
                "opencv-python",
                "scipy",
                "Pillow",
            ]
        },
    }


def install(project):
    import urllib.request

    if any(package(n) is None for n in ["torch", "transformers", "pymeshlab", "mlx"]):
        uv = shutil.which("uv")
        if not uv or not CHECKOUT:
            raise ValueError(
                "Для зависимостей Python выполните uv sync --locked --all-extras в checkout проекта"
            )
        subprocess.run([uv, "sync", "--locked", "--all-extras"], cwd=CHECKOUT, check=True)
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise ValueError(
            "Автоустановка внешних программ поддерживает macOS arm64; задайте BLENDER_BIN и установите выбранный backend вручную"
        )
    TOOLS.mkdir(parents=True, exist_ok=True)
    recipes = [project.prepare.resolved(f.id) for f in project.figures] or [
        project.prepare.defaults
    ]
    need_mv = any(r.shape.method == "mv" for r in recipes)
    need_spirula = any(
        r.shape.method == "spirula" or r.appearance.registration == "orbit" for r in recipes
    )
    need_texture = any(r.appearance.method == "photo" for r in recipes)
    downloads = [
        (
            "blender-4.5.14-macos-arm64.dmg",
            "https://download.blender.org/release/Blender4.5/blender-4.5.14-macos-arm64.dmg",
            "65134d9b07b20e2fa8d3c9e44f6f44ffb5c9774dd521b95f50387310241ca170",
            "blender/Blender.app",
            "Blender.app",
        )
    ]
    if need_spirula:
        downloads.append(
            (
                "spirula-2026.9.30-macos-vulkan-arm64.dmg",
                "https://github.com/harry7557558/spirula-studio/releases/download/v2026.9.30/spirula-2026.9.30-macos-vulkan-arm64.dmg",
                "7a43aa51ca60233129e547198cf0c4efe9f46de7e942980753f959b86a2908cc",
                "Spirula Studio.app",
                "Spirula Studio.app",
            )
        )
    if need_texture:
        downloads.append(
            (
                "OpenMVS_macOS_arm64.zip",
                "https://github.com/cdcseacave/openMVS/releases/download/v2.4.0/OpenMVS_macOS_arm64.zip",
                "3d4c616c97031b1ab6e2eecb0ddd5614fb99513c0782a32c9350602faf38799b",
                "openmvs",
                None,
            )
        )

    def download(url, path, checksum):
        if path.exists() and sha(path) == checksum:
            return
        part = path.with_suffix(".download")
        print(f"Загрузка {path.name}", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response, part.open("wb") as stream:
            shutil.copyfileobj(response, stream)
        if sha(part) != checksum:
            raise ValueError(f"Контрольная сумма не совпадает: {path.name}")
        os.replace(part, path)

    for filename, url, checksum, directory, app_name in downloads:
        dest = TOOLS / directory
        if dest.exists():
            continue
        path = TOOLS / filename
        download(url, path, checksum)
        if app_name:
            mount = TOOLS / "studio-mount"
            mount.mkdir(exist_ok=True)
            subprocess.run(
                [
                    "hdiutil",
                    "attach",
                    "-readonly",
                    "-nobrowse",
                    "-mountpoint",
                    str(mount),
                    str(path),
                ],
                check=True,
            )
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(["ditto", str(mount / app_name), str(dest)], check=True)
            finally:
                subprocess.run(["hdiutil", "detach", str(mount)], check=True)
        else:
            subprocess.run(["unzip", "-q", str(path), "-d", str(dest)], check=True)
    if not shutil.which("ffmpeg"):
        if not shutil.which("brew"):
            raise ValueError("Установите FFmpeg и добавьте его в PATH")
        subprocess.run(["brew", "install", "ffmpeg"], check=True)
    if need_mv:
        source = TOOLS / "hunyuan3d-swift"
        commit = "292331f4d26ddb80b9dcea6bcb5629ff82f12b82"
        if not source.exists():
            subprocess.run(
                ["git", "clone", "https://github.com/ZimengXiong/Hunyuan3D-Swift.git", str(source)],
                check=True,
            )
            subprocess.run(["git", "-C", str(source), "checkout", "--detach", commit], check=True)
        if (
            subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
            ).strip()
            != commit
        ):
            raise ValueError("Версия Hunyuan MLX отличается от зафиксированной")
        meta = json.loads((PACKAGE / "methods/mv/mv-model.json").read_text())
        folder = TOOLS / "hunyuan-mv-turbo"
        folder.mkdir(exist_ok=True)
        shutil.copy2(
            PACKAGE / "methods/mv/mv-config.yaml",
            folder / "config.yaml",
        )
        url = meta.get("url") or meta.get("download_url")
        if not url:
            raise ValueError("Нет URL в зафиксированном описании весов MV")
        download(url, folder / "model.fp16.safetensors", meta["sha256"])
    if any(r.matte.method == "birefnet" for r in recipes):
        from boardgen.video.neuralseg import load_model

        load_model()
    return check(project)
