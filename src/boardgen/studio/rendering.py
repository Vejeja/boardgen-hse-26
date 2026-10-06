from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from boardgen.rendering import blender_render

from . import backgrounds, environment
from .datasets import labels
from .models import Project
from .storage import Artifact, Store, json_write


def camera(spec, rng):
    distance = rng.uniform(*spec.distance_m)
    elevation = rng.uniform(*spec.elevation_deg)
    azimuth = rng.uniform(*spec.azimuth_deg)
    el, az = math.radians(elevation), math.radians(azimuth)
    position = [
        distance * math.cos(el) * math.cos(az) + rng.uniform(-spec.jitter_xy_m, spec.jitter_xy_m),
        distance * math.cos(el) * math.sin(az) + rng.uniform(-spec.jitter_xy_m, spec.jitter_xy_m),
        distance * math.sin(el),
    ]
    return {
        "parameterization": "spherical_board_center_v1",
        "distance_m": distance,
        "elevation_deg": elevation,
        "azimuth_deg": azimuth,
        "position_m": position,
        "roll_deg": rng.uniform(-spec.jitter_roll_deg, spec.jitter_roll_deg),
        "exposure_ev": spec.exposure_ev,
    }


def negative_slots(count, fraction, seed, mixed=False):
    total = round(count * fraction)
    rng = random.Random(seed)
    if not mixed:
        return set(rng.sample(range(count), total))
    slots = set()
    for parity in (0, 1):
        allocation = total // 2 + (total % 2 if parity == seed % 2 else 0)
        slots.update(rng.sample(list(range(parity, count, 2)), allocation))
    return slots


def figure_heights(store, project, assets):
    heights = {}
    measured = {}
    figures = [f for f in project.figures if f.id in assets]
    reference = project.scenes.reference_class
    if reference not in assets:
        reference = figures[0].id if figures else None
    if project.scenes.scale_basis == "shared_video":
        import cv2
        import numpy as np

        for figure in figures:
            frame_id = assets[figure.id]["manifest"]["metadata"].get("frames_id")
            if not frame_id:
                raise ValueError(
                    f"{figure.name}: оценка размера по видео требует подготовленные маски. Подготовьте кадры или задайте измеренные размеры."
                )
            frame_asset = store.artifact(frame_id)
            root = Path(frame_asset["directory"]) / "views"
            data = json.loads((root / "dataset.json").read_text())
            pixel_heights = []
            for index in data["fit"]:
                mask = cv2.imread(str(root / "reference/masks" / f"view_{index:04d}.png"), 0)
                ys = np.where(mask >= 128)[0]
                if len(ys):
                    pixel_heights.append(
                        (ys.max() - ys.min() + 1)
                        * (data["crop_xyxy"][3] - data["crop_xyxy"][1])
                        / mask.shape[0]
                    )
            if not pixel_heights:
                raise ValueError(f"{figure.name}: пустые маски для оценки размера")
            capture = cv2.VideoCapture(data["source_video"])
            frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            capture.release()
            if frame_height <= 0:
                raise ValueError("Не удалось измерить размер исходного кадра")
            measured[figure.id] = float(np.median(pixel_heights) / frame_height)
        reference_height = project.scenes.height_overrides.get(
            reference, project.scenes.nominal_height_m
        )
        heights = {fid: reference_height * px / measured[reference] for fid, px in measured.items()}
    else:
        heights = {
            f.id: project.scenes.nominal_height_m
            * assets[f.id]["manifest"]["metadata"].get("relative_height", 1)
            for f in figures
        }
    heights.update({k: v for k, v in project.scenes.height_overrides.items() if k in assets})
    return heights, {
        "basis": project.scenes.scale_basis,
        "mask_height_fraction_of_frame": measured,
        "height_m": heights,
        "reference": reference,
    }


def sample_objects(assets, requested, width, height, rng, first_class=None):
    margin = 0.05 * min(width, height)
    objects, placed = [], []
    for _ in range(requested):
        cid = first_class if not objects and first_class is not None else rng.choice(list(assets))
        asset = assets[cid]
        yaw = rng.uniform(0, 360)
        radius = max(
            asset["height_m"]
            * asset.get("radius_over_height", math.sqrt(2) * asset["footprint_over_height"] / 2),
            0.01,
        )
        xmin, xmax = -width / 2 + radius + margin, width / 2 - radius - margin
        ymin, ymax = -height / 2 + radius + margin, height / 2 - radius - margin
        if xmin >= xmax or ymin >= ymax:
            continue
        for _ in range(1000):
            x, y = rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)
            if all(
                (x - px) ** 2 + (y - py) ** 2 >= max(0.03, radius + pr) ** 2
                for px, py, pr in placed
            ):
                placed.append((x, y, radius))
                objects.append(
                    {
                        "mesh_path": asset["path"],
                        "height_m": asset["height_m"],
                        "xy": [x, y],
                        "yaw_deg": yaw,
                        "pass_index": len(objects) + 1,
                        "class_id": cid,
                    }
                )
                break
    return objects


def _render_assets(
    store: Store, project: Project, assets: Mapping[str, Artifact]
) -> tuple[dict, dict, dict]:
    by_id = {}
    input_versions = {}
    heights, scale_provenance = figure_heights(store, project, assets)
    for i, figure in enumerate(project.figures):
        item = assets.get(figure.id)
        if not item:
            raise ValueError(
                f"{figure.name}: нет готовой 3D-модели. Подготовьте фигуру или выберите прежний результат."
            )
        manifest = item["manifest"]["metadata"]
        if manifest.get("appearance") == "neutral" and not project.scenes.neutral_ablation:
            raise ValueError(
                f"{figure.name}: модель без текстуры. Выберите фототекстуру или явно разрешите нейтральный материал в дополнительных настройках."
            )
        path = Path(item["directory"]) / manifest["mesh"]
        if path.suffix.lower() != ".glb" or not path.is_file():
            raise ValueError(f"{figure.name}: требуется существующая GLB-модель: {path}")
        if not all(
            math.isfinite(v) and v > 0
            for v in (heights[figure.id], manifest["footprint_over_height"])
        ):
            raise ValueError(f"{figure.name}: некорректный размер 3D-модели")
        by_id[i] = {
            "path": str(path),
            "height_m": heights[figure.id],
            "footprint_over_height": manifest["footprint_over_height"],
        }

        by_id[i]["radius_over_height"] = manifest.get(
            "radius_over_height", math.sqrt(2) * manifest["footprint_over_height"] / 2
        )
        input_versions[figure.id] = item["id"]
    return by_id, input_versions, scale_provenance


def _scene_sources(
    store: Store, project: Project, count: int, index: int, split: str | None
) -> tuple:
    fixed_split = split
    if fixed_split is None:
        train_end = round(
            count * (1 - project.dataset.val_fraction - project.dataset.test_fraction)
        )
        val_end = round(count * (1 - project.dataset.test_fraction))
        fixed_split = "train" if index < train_end else "val" if index < val_end else "test"
    sources = project.scenes.split_sources.get(fixed_split)
    board_sources = sources.boards if sources else project.scenes.boards
    sky_sources = sources.skyboxes if sources else project.scenes.skyboxes
    boards = [(store.resolve(b.image, exists=True), b.width_m, b.height_m) for b in board_sources]
    skyboxes = [store.resolve(p, exists=True) for p in sky_sources] or [None]
    if not boards:
        if project.scenes.scene_type != "platform":
            raise ValueError("Выберите хотя бы одно изображение поля")
        boards = [(None, 0.5, 0.5)]
    kind = project.scenes.scene_type
    if kind == "mixed":
        kind = "board" if index % 2 == 0 else "platform"

    return fixed_split, boards, skyboxes, kind


def _build_scene(
    project: Project,
    destination: Path,
    by_id: dict,
    i: int,
    attempt: int,
    boards: list,
    skyboxes: list,
    kind: str,
    reserved: set[int],
    source_cache: dict,
) -> tuple[dict, bool, Path | None, Path | None]:
    seed = project.scenes.seed + i * 1009 + attempt * 1000003
    rng = random.Random(seed)
    board, width, height = rng.choice(boards)
    skybox = rng.choice(skyboxes)
    for path in (board, skybox):
        if path not in source_cache:
            source_cache[path] = backgrounds.identity(path)
    options = {**camera(project.scenes.camera, rng), "seed": seed}
    negative = rng.random() < project.scenes.negative_fraction
    if project.scenes.stratified_negatives:
        negative = i in reserved
    scene_id = i * project.scenes.max_attempts + attempt
    requested = 0 if negative else rng.randint(*project.scenes.objects_per_scene)
    if kind == "platform" and not negative:
        requested = 1
    first_class = (
        project.scenes.seed + (i // 2 if project.scenes.scene_type == "mixed" else i)
    ) % len(by_id)
    objects = sample_objects(
        by_id,
        requested,
        width,
        height,
        rng,
        first_class=first_class if project.scenes.balanced_classes else None,
    )
    surface_z = 0.0
    platform = None
    if kind == "platform":
        h = objects[0]["height_m"] if objects else project.scenes.nominal_height_m
        radius = by_id[objects[0]["class_id"]]["radius_over_height"] * h if objects else 0
        platform = {
            "radius_m": max(project.scenes.platform.radius_m, radius * 1.15),
            "height_m": project.scenes.platform.height_m,
            "color": [rng.uniform(0.18, 0.5)] * 3,
        }
        surface_z = platform["height_m"]
        fraction = rng.uniform(*project.scenes.platform.frame_fraction)
        distance = h / (2 * math.tan(math.radians(project.scenes.camera.fov_deg) / 2) * fraction)
        cfg = project.scenes.camera.model_copy(
            update={
                "distance_m": (distance, distance),
                "jitter_xy_m": 0,
                "elevation_deg": project.scenes.platform.elevation_deg,
            }
        )
        options = {**camera(cfg, rng), "seed": seed}
        target_z = surface_z + h / 2
        options["position_m"][2] += target_z
        options["target_m"] = [0, 0, target_z]
        for obj in objects:
            obj["xy"] = [0, 0]
    options["exposure_ev"] += rng.uniform(
        -project.scenes.exposure_jitter_ev, project.scenes.exposure_jitter_ev
    )
    name = f"raw_{scene_id:06d}"
    image = destination / "images/raw" / f"{name}.png"
    scene = {
        "schema": blender_render.SCHEMA,
        "output": {
            "rgb_path": str(image.resolve()),
            "id_path": str((destination / "tmp" / f"{name}_id.exr").resolve()),
        },
        "board": {
            "image": str(board) if board else None,
            "width_m": width,
            "height_m": height,
        },
        "scene_type": kind,
        "platform": platform,
        "env": {
            "hdri": str(skybox) if skybox else None,
            "strength": project.scenes.environment_strength,
            "rotation_deg": rng.uniform(*project.scenes.environment_rotation_deg),
        },
        "lighting": {
            "energy": rng.uniform(*project.scenes.light_energy),
            "azimuth_deg": rng.uniform(-180, 180),
        },
        "camera": {**options, "fov_deg": project.scenes.camera.fov_deg},
        "objects": objects,
        "image_size": list(project.scenes.image_size),
        "seed": seed,
        "flags": {
            "requested_objects": requested,
            "mesh_hover_m": surface_z + 0.00005,
            "render_samples": project.scenes.render_samples,
            "id_encoding": blender_render.ID_ENCODING,
        },
    }
    return scene, negative, board, skybox


def _valid_labels(rows: list, scene: dict, negative: bool, min_side_px: int) -> bool:
    wpx, hpx = scene["image_size"]
    valid = (negative and not rows) or (
        bool(rows)
        and len(rows) == len(scene["objects"])
        and len(rows) == scene["flags"]["requested_objects"]
        and all(min(r[3] * wpx, r[4] * hpx) >= min_side_px for r in rows)
    )
    return valid


def render(
    store: Store,
    project: Project,
    destination: Path,
    assets: Mapping[str, Artifact],
    count: int,
    *,
    split: str | None = None,
    session: blender_render.BlenderSession | None = None,
) -> list[dict[str, Any]]:
    """Сформировать сцены и принять только разметку всех запрошенных видимых объектов.

    При неудачной попытке используется следующий детерминированный seed.
    Источники фона выбираются из настроек соответствующего раздела датасета.
    """
    by_id, input_versions, scale_provenance = _render_assets(store, project, assets)
    records = []
    source_cache = {None: backgrounds.identity(None)}
    reserved = negative_slots(
        count,
        project.scenes.negative_fraction,
        project.scenes.seed,
        project.scenes.scene_type == "mixed",
    )
    for i in range(count):
        fixed_split, boards, skyboxes, kind = _scene_sources(store, project, count, i, split)

        accepted = False
        for attempt in range(project.scenes.max_attempts):
            scene, negative, board, skybox = _build_scene(
                project,
                destination,
                by_id,
                i,
                attempt,
                boards,
                skyboxes,
                kind,
                reserved,
                source_cache,
            )
            name = f"raw_{i * project.scenes.max_attempts + attempt:06d}"
            image = destination / "images/raw" / f"{name}.png"
            label = destination / "labels/raw" / f"{name}.txt"
            blender_render.render_scene(
                scene,
                destination / "tmp" / f"{name}.json",
                label,
                blender_bin=environment.blender(),
                session=session,
            )

            rows = labels(label, len(project.figures), allow_empty=True)
            valid = _valid_labels(rows, scene, negative, project.scenes.min_side_px)
            if valid:
                record = {
                    "image": str(image),
                    "labels": rows,
                    "group": f"scene:{project.scenes.seed}:{i}",
                    "origin": "synthetic",
                    "scene": scene,
                    "seed": scene["seed"],
                    "attempt": attempt,
                    "assets": input_versions,
                    "figure_scale": scale_provenance,
                    "board_sha256": source_cache[board]["sha256"],
                    "skybox_sha256": source_cache[skybox]["sha256"],
                    "board_source_id": source_cache[board]["source_id"],
                    "skybox_source_id": source_cache[skybox]["source_id"],
                    "scene_type": kind,
                    "fixed_split": fixed_split,
                }
                records.append(record)
                accepted = True
                print(f"Сцены: {i + 1}/{count}", flush=True)
                break
            print(f"Сцена {i}: повтор {attempt + 1}, фигуры не видны или слишком малы", flush=True)
        if not accepted:
            raise ValueError(
                f"Сцена {i}: после {project.scenes.max_attempts} попыток не удалось получить корректную разметку. Измените камеру, размеры фигур или число объектов."
            )
    json_write(destination / "scenes.json", records)
    return records
