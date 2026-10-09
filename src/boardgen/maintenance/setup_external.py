import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.request

from boardgen.studio.environment import REPOSITORY as PROJECT

TOOLS = PROJECT / "tmp/benchmark_tools"


def run(*args):
    subprocess.run([str(a) for a in args], cwd=PROJECT, check=True)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def download(url, target, sha):
    if not target.exists() or digest(target) != sha:
        partial = target.with_suffix(target.suffix + ".download")
        with urllib.request.urlopen(url, timeout=60) as source, partial.open("wb") as output:
            shutil.copyfileobj(source, output)
        if digest(partial) != sha:
            raise ValueError(f"SHA256 mismatch: {target.name}")
        partial.replace(target)


def app(dmg, destination, name):
    if destination.exists():
        return
    mount = TOOLS / "setup-mount"
    mount.mkdir(exist_ok=True)
    run("hdiutil", "attach", "-readonly", "-nobrowse", "-mountpoint", mount, dmg)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        run("ditto", mount / name, destination)
    finally:
        run("hdiutil", "detach", mount)


def main():
    parser = argparse.ArgumentParser(
        description="Install pinned macOS arm64 tools and reconstruction weights into ignored tmp/."
    )
    parser.add_argument(
        "--weights", action="store_true", help="also download the pinned Hunyuan MV checkpoint"
    )
    args = parser.parse_args()
    import platform

    if (sys.platform, platform.machine()) != ("darwin", "arm64"):
        raise SystemExit(
            "This external-tool recipe is for macOS arm64; CPU Python extras are portable."
        )
    TOOLS.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, url, sha, destination, app_name in [
        (
            "blender-4.5.14-macos-arm64.dmg",
            "https://download.blender.org/release/Blender4.5/blender-4.5.14-macos-arm64.dmg",
            "65134d9b07b20e2fa8d3c9e44f6f44ffb5c9774dd521b95f50387310241ca170",
            "blender/Blender.app",
            "Blender.app",
        ),
        (
            "spirula-2026.9.30-macos-vulkan-arm64.dmg",
            "https://github.com/harry7557558/spirula-studio/releases/download/v2026.9.30/spirula-2026.9.30-macos-vulkan-arm64.dmg",
            "7a43aa51ca60233129e547198cf0c4efe9f46de7e942980753f959b86a2908cc",
            "Spirula Studio.app",
            "Spirula Studio.app",
        ),
    ]:
        download(url, TOOLS / filename, sha)
        app(TOOLS / filename, TOOLS / destination, app_name)
        records.append({"file": filename, "url": url, "sha256": sha, "app": destination})
    zip_name = "OpenMVS_macOS_arm64.zip"
    zip_sha = "3d4c616c97031b1ab6e2eecb0ddd5614fb99513c0782a32c9350602faf38799b"
    zip_url = (
        "https://github.com/cdcseacave/openMVS/releases/download/v2.4.0/OpenMVS_macOS_arm64.zip"
    )
    download(zip_url, TOOLS / zip_name, zip_sha)
    if not (TOOLS / "openmvs/DensifyPointCloud").exists():
        run("unzip", "-q", TOOLS / zip_name, "-d", TOOLS / "openmvs")
    records.append({"file": zip_name, "url": zip_url, "sha256": zip_sha})
    for directory, url, commit in [
        (
            "hunyuan3d-swift",
            "https://github.com/ZimengXiong/Hunyuan3D-Swift.git",
            "292331f4d26ddb80b9dcea6bcb5629ff82f12b82",
        ),
    ]:
        target = TOOLS / directory
        if not target.exists():
            run("git", "clone", url, target)
            run("git", "-C", target, "checkout", "--detach", commit)
        actual = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != commit:
            raise ValueError(f"Unexpected source revision: {directory}: {actual}")
        records.append({"source": directory, "url": url, "commit": actual})
    folder = TOOLS / "hunyuan-mv-turbo"
    folder.mkdir(exist_ok=True)
    shutil.copy2(PROJECT / "src/boardgen/methods/mv/mv-config.yaml", folder / "config.yaml")
    if args.weights:
        run(
            sys.executable,
            "-m",
            "boardgen.methods.mv.download_weights",
            PROJECT / "src/boardgen/methods/mv/mv-model.json",
            folder,
        )
    output = PROJECT / "out/setup/external-tools.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2))
    print(f"External prerequisites ready: {output}")


if __name__ == "__main__":
    main()
