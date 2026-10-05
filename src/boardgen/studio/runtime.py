import os
import subprocess
import sys

from . import environment


def command(module, arguments):
    argv = [sys.executable, "-u", "-m", module, *map(str, arguments)]
    print("Запуск:", " ".join(argv), flush=True)
    subprocess.run(
        argv,
        cwd=environment.REPOSITORY,
        check=True,
        env={
            **os.environ,
            "BOARDGEN_TOOLS_DIR": str(environment.TOOLS),
            "SPIRULA_BIN": str(environment.spirula()),
        },
    )


def work(folder, name):
    p = folder / name
    p.mkdir(parents=True, exist_ok=False)
    return p


def stage_versions(versions, stage):
    modules = {
        "cameras": {
            "methods/orbit/calibrate_orbit.py",
            "camera/camera_audit.py",
            "evaluation/benchmark_tools.py",
        },
        "geometry": {
            "methods/mv/run_mv.py",
            "methods/mv/mv_adapter.py",
            "methods/mv/package_generated.py",
            "methods/orbit/run_candidates.py",
            "methods/orbit/constrain_surface.py",
            "methods/orbit/geometry_proxy.py",
            "methods/orbit/calibrate_orbit.py",
            "camera/camera_audit.py",
            "evaluation/benchmark_tools.py",
        },
        "fusion": {
            "methods/orbit/depth_fusion.py",
            "methods/orbit/regularize_depth.py",
            "methods/orbit/constrain_surface.py",
            "methods/orbit/run_candidates.py",
        },
        "appearance": {
            "methods/mv/texture_mv.py",
            "methods/mv/blend_mv_texture.py",
            "methods/orbit/refit_mv_texture.py",
            "studio/texture.py",
            "camera/camera_audit.py",
        },
    }[stage]
    return {
        "code": {k: v for k, v in versions["code"].items() if k in modules},
        "tools": versions["tools"],
        "external_code": versions.get("external_code", {}) if stage == "geometry" else {},
        "packages": versions["packages"],
    }
