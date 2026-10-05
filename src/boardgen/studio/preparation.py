import json
import shutil
import subprocess
from contextvars import ContextVar
from pathlib import Path

from . import environment
from .runtime import command, stage_versions, work
from .storage import digest, json_write, sha

series_inputs = ContextVar("series_inputs", default=None)


def input_cached(store, key, reuse_completed):
    shared = series_inputs.get()
    if shared is not None and key in shared:
        return store.artifact(shared[key])
    item = store.cached(key) if reuse_completed else None
    if item and shared is not None:
        shared[key] = item["id"]
    return item


def remember_input(item):
    shared = series_inputs.get()
    if shared is not None:
        shared[item["cache_key"]] = item["id"]
    return item


def frame_key(store, figure, recipe, versions):
    return digest(
        {
            "stage": "frames.v1",
            "figure_id": figure.id,
            "source": sha(store.resolve(figure.video, exists=True)),
            "clip": figure.clip.model_dump(),
            "matte": recipe.matte.model_dump(),
            "frames": recipe.shape.candidate_frames,
            "size": recipe.shape.frame_size,
            "versions": {
                "code": {
                    k: v
                    for k, v in versions["code"].items()
                    if k
                    in {
                        "video/views.py",
                        "common/manifest.py",
                        "video/neuralseg.py",
                        "video/segment.py",
                        "video/turntable.py",
                    }
                },
                "neural": versions["neural"],
                "packages": {
                    k: v
                    for k, v in versions["packages"].items()
                    if k in {"torch", "transformers", "opencv-python", "numpy", "Pillow", "scipy"}
                },
            },
        }
    )


def frames(store, project, figure, folder, versions):
    from boardgen.video.views import prepare_views

    recipe = project.prepare.resolved(figure.id)
    key = frame_key(store, figure, recipe, versions)
    cached = input_cached(store, key, project.prepare.reuse_completed)
    if cached:
        print(f"{figure.name}: проверенный кэш кадров", flush=True)
        return cached
    temporary = work(folder, f"frames-{figure.id}")
    video = store.resolve(figure.video, exists=True)
    source_hash = sha(video)
    selected_video = video
    if figure.clip.start_s or figure.clip.end_s is not None:
        selected_video = temporary / "clip.mp4"
        cmd = ["ffmpeg", "-v", "error", "-ss", str(figure.clip.start_s), "-i", str(video)]
        if figure.clip.end_s is not None:
            cmd.extend(["-t", str(figure.clip.end_s - figure.clip.start_s)])
        subprocess.run(
            [*cmd, "-an", "-c:v", "libx264", "-crf", "0", str(selected_video)], check=True
        )
    data_root = temporary / "views"
    prepare_views(
        data_root,
        selected_video,
        recipe.shape.frame_size,
        recipe.shape.candidate_frames,
        full_clip=True,
        matte_method=recipe.matte.method,
        mask_size=recipe.matte.input_size,
        cross_check=recipe.matte.cross_check,
    )
    data = json.loads((data_root / "dataset.json").read_text())
    if any(d["empty"] for d in data["diagnostics"]):
        raise ValueError(
            f"{figure.name}: есть пустые маски. Измените обработку или фрагмент видео."
        )
    return remember_input(
        store.publish(
            temporary,
            "frames",
            key,
            {
                "figure_id": figure.id,
                "source_sha256": source_hash,
                "source": str(video),
                "clip": figure.clip.model_dump(),
                "recipe": {
                    "matte": recipe.matte.model_dump(),
                    "candidate_frames": recipe.shape.candidate_frames,
                    "frame_size": recipe.shape.frame_size,
                },
                "dataset": "views/dataset.json",
                "contact": "views/contact.png",
                "fit": data["fit"],
                "heldout": data["heldout"],
                "diagnostics": data["diagnostics"],
                "versions": versions,
            },
            figure.id,
        )
    )


def asset_metadata(mesh_path, figure, recipe, frame_asset, versions):
    import numpy as np
    import trimesh

    mesh = trimesh.load(mesh_path, force="scene", process=False).to_geometry()
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("Некорректная 3D-модель")
    extent = mesh.extents
    if extent[1] <= 0:
        raise ValueError("Модель не имеет высоты Y")
    return {
        "figure_id": figure.id,
        "name": figure.name,
        "mesh": "figure.glb",
        "faces": len(mesh.faces),
        "up_axis": "Y",
        "footprint_over_height": float(max(extent[0], extent[2]) / extent[1]),
        "appearance": recipe.appearance.method,
        "source_sha256": frame_asset["manifest"]["metadata"]["source_sha256"],
        "frames_id": frame_asset["id"],
        "recipe": recipe.model_dump(),
        "reviewed_views": figure.reviewed_views,
        "versions": versions,
        "quality_state": "candidate_requires_review",
    }


def camera_stage(store, project, figure, frame_asset, folder, versions, method):
    from boardgen.evaluation.benchmark_tools import sfm

    key = digest(
        {
            "stage": "cameras.v1",
            "figure": figure.id,
            "frames": frame_asset["cache_key"],
            "method": method,
            "front": figure.front_view,
            "period": figure.period_views,
            "focal": figure.focal_px,
            "versions": stage_versions(versions, "cameras"),
        }
    )
    cached = input_cached(store, key, project.prepare.reuse_completed)
    if cached:
        print(f"{figure.name}: кэш камер", flush=True)
        return cached
    temporary = work(folder, f"cameras-{figure.id}")
    root = temporary / "dataset"
    shutil.copytree(Path(frame_asset["directory"]) / "views", root)
    sfm(root, environment.spirula(), name="sfm-sift", features="sift", max_image_size=1024)
    if method == "orbit":
        root = temporary / "orbit"
        period = figure.period_views
        command(
            "boardgen.methods.orbit.calibrate_orbit",
            [
                "custom",
                "--dataset",
                temporary / "dataset",
                "--front-view",
                figure.front_view,
                "--period-range",
                max(5, period - 6),
                period + 6,
                "--focal-px",
                figure.focal_px,
                "--output",
                root,
            ],
        )

    return remember_input(
        store.publish(
            temporary,
            "cameras",
            key,
            {
                "root": str(root.relative_to(temporary)),
                "frames_id": frame_asset["id"],
                "method": method,
                "front_view": figure.front_view,
                "period_views": figure.period_views,
                "focal_px": figure.focal_px,
                "versions": stage_versions(versions, "cameras"),
            },
            figure.id,
        )
    )


def four_views(figure, root):
    return {
        "root": root.name,
        "selection": "User reviewed fit photographs",
        "views": [
            {
                "tag": tag,
                "view": figure.reviewed_views[tag],
                "nominal_angle": angle,
                "angle_source": "manual approximate quarter turn",
            }
            for tag, angle in [("front", 0), ("left", 90), ("back", 180), ("right", 270)]
        ],
    }


def geometry_stage(store, project, figure, frame_asset, folder, versions):
    recipe = project.prepare.resolved(figure.id)
    if recipe.shape.extraction == "depth_fusion":
        return fusion_stage(store, project, figure, frame_asset, folder, versions)
    key = digest(
        {
            "stage": "geometry.v1",
            "figure": figure.id,
            "frames": frame_asset["cache_key"],
            "shape": recipe.shape.model_dump(),
            "views": figure.reviewed_views,
            "front": figure.front_view,
            "period": figure.period_views,
            "focal": figure.focal_px,
            "native_texture_size": recipe.appearance.texture_size
            if recipe.shape.method == "spirula"
            else None,
            "versions": stage_versions(versions, "geometry"),
        }
    )

    cached = (
        input_cached(store, key, project.prepare.reuse_completed)
        if recipe.shape.method == "spirula"
        else store.cached(key)
        if project.prepare.reuse_completed
        else None
    )
    if cached:
        print(f"{figure.name}: кэш 3D-формы", flush=True)
        return cached
    camera_asset = None
    if recipe.shape.method == "spirula":
        camera_asset = camera_stage(
            store, project, figure, frame_asset, folder, versions, recipe.shape.cameras
        )
    temporary = work(folder, f"geometry-{figure.id}")
    root = temporary / "dataset"
    source_root = (
        Path(camera_asset["directory"]) / camera_asset["manifest"]["metadata"]["root"]
        if camera_asset
        else Path(frame_asset["directory"]) / "views"
    )
    shutil.copytree(source_root, root)
    if recipe.shape.method == "mv":
        selection = four_views(figure, root)
        json_write(temporary / "views.json", selection)
        command(
            "boardgen.methods.mv.run_mv",
            [
                root,
                temporary / "views.json",
                temporary / "mv",
                "--resolution",
                recipe.shape.mesh_resolution,
                "--seed",
                recipe.shape.seed,
                "--seconds",
                recipe.shape.timeout_seconds,
                "--weights",
                environment.TOOLS / "hunyuan-mv-turbo",
                "--manifest",
                environment.PACKAGE / "methods/mv/mv-model.json",
            ],
        )
        from boardgen.methods.mv.package_generated import package

        package(temporary / "mv/neutral.glb", temporary / "geometry", faces=120000)
        mesh = temporary / "geometry.glb"
    else:
        from boardgen.methods.orbit.run_candidates import run

        run(
            "custom",
            recipe.shape.profile,
            recipe.shape.iterations,
            recipe.shape.cap,
            dataset=root,
            texture_size=recipe.appearance.texture_size,
            timeout=recipe.shape.timeout_seconds,
            executable=environment.spirula(),
        )
        mesh = (
            root
            / "runs"
            / f"spirula-{recipe.shape.profile}-{recipe.shape.iterations}"
            / "review/photo.glb"
        )
        shutil.copy2(mesh.parent / "asset.json", temporary / "asset.json")
    shutil.copy2(mesh, temporary / "figure.glb")
    meta = asset_metadata(temporary / "figure.glb", figure, recipe, frame_asset, versions)
    meta.update(
        appearance="neutral" if recipe.shape.method == "mv" else "native",
        camera_id=camera_asset["id"] if camera_asset else None,
    )
    result = store.publish(temporary, "geometry", key, meta, figure.id)
    return remember_input(result) if recipe.shape.method == "spirula" else result


def texture_views(figure, root, count):
    import numpy as np

    selection = four_views(figure, root)
    if count == 4:
        return selection
    indices = np.array([v["view"] for v in selection["views"]], dtype=float)
    if not (np.diff(indices) > 0).all():
        raise ValueError(
            "Для дополнительных фоторакурсов выберите четыре стороны последовательно в одном обороте либо используйте 4 фото"
        )
    angles = np.array([0, 90, 180, 270], dtype=float)
    data = json.loads((root / "dataset.json").read_text())
    wanted = np.linspace(-60, 270, count)
    result, used = [], set()
    for angle in wanted:
        index = (
            np.interp(angle, angles, indices)
            if angle >= 0
            else indices[0] + angle * (indices[1] - indices[0]) / 90
        )
        available = [i for i in data["fit"] if i not in used]
        if not available:
            raise ValueError("Недостаточно fit-кадров для текстуры")
        chosen = min(available, key=lambda i: abs(i - index))
        used.add(chosen)
        nominal = (
            float(np.interp(chosen, indices, angles))
            if chosen >= indices[0]
            else float((chosen - indices[0]) * 90 / (indices[1] - indices[0]))
        )
        result.append(
            {
                "tag": f"photo-{chosen}",
                "view": chosen,
                "nominal_angle": nominal,
                "angle_source": "interpolation/extrapolation of reviewed quarter-turn anchors",
            }
        )
    return {"root": root.name, "selection": "Fit-only interpolated photo anchors", "views": result}


def prepare(store, project, figure, folder, versions):
    recipe = project.prepare.resolved(figure.id)
    frame_asset = frames(store, project, figure, folder, versions)
    if recipe.shape.method == "mv" and figure.reviewed_frames_id != frame_asset["id"]:
        old = store.artifact(figure.reviewed_frames_id) if figure.reviewed_frames_id else None
        reviewed = old and old["kind"] == "frames" and old["figure_id"] == figure.id
        for index in figure.reviewed_views.values():
            name = f"views/train/images/view_{index:04d}.png"
            reviewed = reviewed and old["manifest"]["files"].get(name) == frame_asset["manifest"][
                "files"
            ].get(name)
        if not reviewed:
            raise ValueError(f"{figure.name}: кадры изменились; проверьте четыре ракурса заново")
    if recipe.matte.method == "birefnet":
        from boardgen.video.neuralseg import unload_model

        unload_model()
    geometry = geometry_stage(store, project, figure, frame_asset, folder, versions)
    key = digest(
        {
            "stage": "appearance.v1",
            "geometry": geometry["cache_key"],
            "appearance": recipe.appearance.model_dump(),
            "front": figure.front_view,
            "period": figure.period_views,
            "focal": figure.focal_px,
            "versions": stage_versions(versions, "appearance"),
        }
    )
    if project.prepare.reuse_completed:
        cached = store.cached(key)
        if cached:
            print(f"{figure.name}: кэш текстуры", flush=True)
            return cached
    camera_asset = None
    if (
        recipe.shape.method == "mv"
        and recipe.appearance.method == "photo"
        and recipe.appearance.registration == "orbit"
    ):
        camera_asset = camera_stage(store, project, figure, frame_asset, folder, versions, "orbit")
    temporary = work(folder, f"appearance-{figure.id}")
    source = Path(geometry["directory"]) / "figure.glb"
    mesh = source
    if recipe.appearance.method == "photo":
        root = Path(
            geometry["manifest"]["metadata"].get(
                "dataset_root", str(Path(geometry["directory"]) / "dataset")
            )
        )
        if recipe.shape.method == "mv":
            json_write(
                temporary / "views.json", texture_views(figure, root, recipe.appearance.photo_views)
            )
            command(
                "boardgen.methods.mv.texture_mv",
                [
                    "custom",
                    "--dataset",
                    root,
                    "--source",
                    source,
                    "--views",
                    temporary / "views.json",
                    "--output",
                    temporary / "texture",
                    "--texture-size",
                    recipe.appearance.texture_size,
                    "--tools",
                    environment.TOOLS / "openmvs",
                    *(["--focal-px", figure.focal_px] if figure.focal_px else []),
                ],
            )
            mesh = temporary / "texture/photo.glb"
            if camera_asset:
                calibrated_root = (
                    Path(camera_asset["directory"]) / camera_asset["manifest"]["metadata"]["root"]
                )
                command(
                    "boardgen.methods.orbit.refit_mv_texture",
                    [
                        "--old",
                        temporary / "texture",
                        "--dataset",
                        calibrated_root,
                        "--output",
                        temporary / "registered",
                    ],
                )
                command("boardgen.methods.mv.blend_mv_texture", [temporary / "registered"])
                mesh = temporary / "registered/photo-blended.glb"
        else:
            from .texture import bake_reconstruction

            mesh = bake_reconstruction(
                root, source, temporary / "texture", recipe.appearance.texture_size
            )
    elif recipe.appearance.method == "neutral" and recipe.shape.method == "spirula":
        import trimesh

        model = trimesh.load(source, force="mesh")
        model.visual = trimesh.visual.ColorVisuals(mesh=model, vertex_colors=[160, 160, 160, 255])
        model.export(temporary / "neutral.glb")
        mesh = temporary / "neutral.glb"
    shutil.copy2(mesh, temporary / "figure.glb")
    metadata = asset_metadata(temporary / "figure.glb", figure, recipe, frame_asset, versions)
    metadata["geometry_id"] = geometry["id"]
    metadata["camera_id"] = (
        camera_asset["id"] if camera_asset else geometry["manifest"]["metadata"].get("camera_id")
    )
    for name in ("asset.json", "registration.json"):
        origin = Path(geometry["directory"]) / name
        if origin.is_file():
            shutil.copy2(origin, temporary / name)
    if (temporary / "registered/registration.json").is_file():
        shutil.copy2(temporary / "registered/registration.json", temporary / "registration.json")
    if (temporary / "texture/cameras.json").exists():
        cams = json.loads((temporary / "texture/cameras.json").read_text())
        metadata["photo_registration"] = {
            "mean_silhouette_iou": sum(c["silhouette_iou"] for c in cams) / len(cams),
            "views": len(cams),
            "intrinsics": "SfM" if camera_asset else "user_focal_or_nominal_focal",
            "meaning": "silhouette fit diagnostic; not independent reconstruction accuracy",
        }
    return store.publish(temporary, "figure", key, metadata, figure.id)


def fusion_stage(store, project, figure, frame_asset, folder, versions):
    from boardgen.methods.orbit.constrain_surface import constrain
    from boardgen.methods.orbit.run_candidates import export

    from .models import Recipe

    recipe = project.prepare.resolved(figure.id)
    native_project = project.model_copy(deep=True)
    native_recipe = recipe.model_dump()
    native_recipe["shape"]["extraction"] = "native"
    native_recipe["appearance"]["method"] = "native"
    native_project.prepare.overrides[figure.id] = Recipe.model_validate(native_recipe).model_dump()
    native = geometry_stage(store, native_project, figure, frame_asset, folder, versions)
    key = digest(
        {
            "stage": "depth-fusion.v1",
            "native": native["cache_key"],
            "representation_id": native["id"],
            "versions": stage_versions(versions, "fusion"),
        }
    )
    if project.prepare.reuse_completed:
        cached = store.cached(key)
        if cached:
            return cached
    temporary = work(folder, f"fusion-{figure.id}")
    root = Path(native["directory"]) / "dataset"
    run = root / "runs" / f"spirula-{recipe.shape.profile}-{recipe.shape.iterations}"
    raw, regularized = temporary / "raw", temporary / "consistent"
    command(
        "boardgen.methods.orbit.depth_fusion",
        [root, run, "--executable", environment.spirula(), "--output-directory", raw],
    )
    command(
        "boardgen.methods.orbit.regularize_depth",
        [root, run, "--source", raw, "--output-directory", regularized],
    )
    constrain(root, regularized / "mesh.glb", temporary / "constrained.glb")
    export(root, temporary / "constrained.glb", temporary / "review")
    shutil.copy2(temporary / "review/photo.glb", temporary / "figure.glb")
    shutil.copy2(temporary / "review/asset.json", temporary / "asset.json")
    metadata = asset_metadata(temporary / "figure.glb", figure, recipe, frame_asset, versions)
    metadata.update(
        appearance="neutral",
        representation_id=native["id"],
        dataset_root=str(root),
        camera_id=native["manifest"]["metadata"]["camera_id"],
    )
    return store.publish(temporary, "geometry", key, metadata, figure.id)
