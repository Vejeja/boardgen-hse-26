from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

import numpy as np

from boardgen.camera.camera_audit import read_model
from boardgen.evaluation.figure_comparison import camera_matrices, model_to_world

from . import environment
from .selection import available_packages
from .storage import digest, sha


def packages():
    return {digest(str(p))[:16]: p for p in available_packages(environment.REPOSITORY)}


def package_path(identifier):
    if identifier not in packages():
        raise ValueError("Набор сравнения отсутствует")
    return packages()[identifier]


def package_file(identifier, filename):
    package = package_path(identifier)
    allowed = json.loads((package / "gallery/comparison-assets.json").read_text())
    if filename not in allowed:
        raise ValueError("Файл отсутствует в реестре сравнения")
    path = (package / "gallery" / filename).resolve()
    if not path.is_relative_to(package) or not path.is_file():
        raise ValueError("Файл выходит за пределы набора или отсутствует")
    return path


def package_figure(identifier, object_id):
    package = package_path(identifier)
    catalog = json.loads((package / "gallery/comparison-catalog.json").read_text())
    row = next((r for r in catalog if r["object"] == object_id), None)
    if not row:
        raise ValueError("Фигура отсутствует в наборе")
    path = (package / "gallery" / row["file"]).resolve()
    if not path.is_relative_to(package / "gallery"):
        raise ValueError("Некорректный каталог")
    data = json.loads(path.read_text())
    for frame in data["frames"]:
        for key in ("mask", "photo", "contour"):
            frame[key] = file_url(identifier, frame[key])
    for item in data["candidates"]:
        item["mesh"] = file_url(identifier, item["mesh"])
    return data


def file_url(identifier, filename):
    return f"/api/comparison/file/{identifier}/{quote(filename, safe='')}"


def catalog(store, identifier="project"):
    if identifier != "project":
        return json.loads(
            (package_path(identifier) / "gallery/comparison-catalog.json").read_text()
        )
    return [{"object": f.id, "name": f.name} for f in store.load().figures]


def artifact_url(item, name):
    if name not in item["manifest"]["files"]:
        raise ValueError(f"Файл отсутствует в версии: {name}")
    return f"/api/artifacts/{item['id']}/{quote(name, safe='/')}"


def candidate_camera(store, item):
    meta = item["manifest"]["metadata"]
    identifier = meta.get("camera_id")
    if not identifier and meta.get("geometry_id"):
        identifier = store.artifact(meta["geometry_id"], verify=False)["manifest"]["metadata"].get(
            "camera_id"
        )
    return store.artifact(identifier, verify=False) if identifier else None


def candidate_transform(item):
    root = Path(item["directory"])
    if (root / "asset.json").is_file():
        return model_to_world(json.loads((root / "asset.json").read_text())["transform"])
    if (root / "registration.json").is_file():
        from scipy.spatial.transform import Rotation

        x = json.loads((root / "registration.json").read_text())["parameters"]
        transform = np.eye(4)
        transform[:3, :3] = (
            np.exp(x[0]) * Rotation.from_euler("xyz", x[1:4], degrees=True).as_matrix()
        )
        transform[:3, 3] = x[4:7]
        model_to_world(transform)
        return transform
    raise ValueError("Нет сохранённого преобразования в систему камер; доступен отдельный просмотр")


def project_figure(store, object_id):
    import trimesh

    project = store.load()
    figure = next((f for f in project.figures if f.id == object_id), None)
    if not figure:
        raise ValueError("Фигура отсутствует в проекте")
    source_hash = sha(store.resolve(figure.video, exists=True))
    items = [
        a
        for a in store.artifacts("figure")
        if a["figure_id"] == object_id
        and a["manifest"]["metadata"].get("source_sha256") == source_hash
    ]
    selected = project.scenes.asset_versions.get(object_id)
    items.sort(key=lambda a: a["id"] != selected)
    chosen_camera = None
    entries, unavailable = [], []
    bounds = None
    for item in items:
        try:
            camera = candidate_camera(store, item)
            if not camera:
                raise ValueError("Нет сохранённых камер")
            root = Path(camera["directory"]) / camera["manifest"]["metadata"]["root"]
            hashes = [sha(root / "train/sparse/0" / n) for n in ("cameras.bin", "images.bin")]
            transform = candidate_transform(item)
            if chosen_camera and hashes != chosen_camera[2]:
                raise ValueError("Другая система камер; сравнивайте отдельно")
            chosen_camera = chosen_camera or (camera, root, hashes)
            meta = item["manifest"]["metadata"]
            recipe = meta.get("recipe") or {}
            method = recipe.get("shape", {}).get("method", meta.get("method", "3D"))
            extraction = recipe.get("shape", {}).get("extraction", "native")
            label = meta.get("research_title") or (
                "MV: генеративная форма"
                if method == "mv"
                else f"Spirula: {'depth fusion' if extraction == 'depth_fusion' else 'native'}"
            )
            mesh = meta.get("mesh", "figure.glb")
            entries.append(
                {
                    "id": item["id"],
                    "label": label,
                    "mesh": artifact_url(item, mesh),
                    "mesh_sha256": item["manifest"]["files"][mesh],
                    "world_from_model": transform.tolist(),
                    "note": meta.get("research_note", "Версия требует визуального выбора"),
                    "metrics": None,
                    "method": "hunyuan_mv" if method == "mv" else "spirula",
                    "default": len(entries) < 3 or item["id"] == selected,
                    "download_name": f"{object_id}-{item['id'][:8]}.glb",
                }
            )
            if bounds is None:
                geometry = trimesh.load(
                    Path(item["directory"]) / mesh, force="scene", process=False
                ).to_geometry()
                geometry.apply_transform(transform)
                bounds = geometry.bounds.tolist()
        except (ValueError, OSError, KeyError) as error:
            meta = item["manifest"]["metadata"]
            mesh = meta.get("mesh", "figure.glb")
            unavailable.append(
                {
                    "id": item["id"],
                    "reason": str(error),
                    "label": meta.get("research_title") or "3D-модель",
                    "mesh": artifact_url(item, mesh),
                    "mesh_sha256": item["manifest"]["files"][mesh],
                }
            )
    if not entries:
        combined = None
        matched_ids = set()
        frame_signature = None
        for package_id in packages():
            try:
                data = package_figure(package_id, object_id)
                if data["source_video_sha256"] != source_hash:
                    continue
                signature = digest(
                    [
                        {k: f[k] for k in ("source_frame", "projection", "view")}
                        for f in data["frames"]
                    ]
                )
                if frame_signature and signature != frame_signature:
                    continue
                matched = []
                for candidate in data["candidates"]:
                    for item in items:
                        mesh = item["manifest"]["metadata"].get("mesh", "figure.glb")
                        if (
                            item["id"] not in matched_ids
                            and item["manifest"]["files"].get(mesh) == candidate["mesh_sha256"]
                            and item["manifest"]["metadata"]
                            .get("selection", {})
                            .get("id", candidate["id"])
                            == candidate["id"]
                        ):
                            matched_ids.add(item["id"])
                            matched.append(
                                {
                                    **candidate,
                                    "id": item["id"],
                                    "mesh": artifact_url(item, mesh),
                                    "default": candidate.get("default", False)
                                    or item["id"] == selected,
                                }
                            )
                if matched:
                    if combined is None:
                        combined = {**data, "candidates": [], "name": figure.name}
                        frame_signature = signature
                    combined["candidates"].extend(matched)
            except (ValueError, OSError, KeyError):
                continue
        remaining = [u for u in unavailable if u["id"] not in matched_ids]
        if combined:
            combined["unavailable"] = remaining
            return combined
        return {
            "object": object_id,
            "name": figure.name,
            "source_video_sha256": source_hash,
            "frames": [],
            "candidates": [],
            "unavailable": remaining,
            "reference_bounds": None,
        }
    camera_asset, root, _ = chosen_camera
    cameras, images, _ = read_model(root / "train/sparse/0")
    dataset = json.loads((root / "dataset.json").read_text())
    frames = []
    for im in sorted(images, key=lambda x: x["name"]):
        c = cameras[im["cid"]]
        name = im["name"]
        index = int(Path(name).stem.split("_")[-1])
        projection, view = camera_matrices(c, im["r"], im["t"])
        prefix = str(root.relative_to(Path(camera_asset["directory"])))
        mask_name = f"{prefix}/reference/masks/{name}"
        frames.append(
            {
                "index": index,
                "source_frame": dataset["source_frame_indices"][index],
                "seconds": None,
                "split": "fit",
                "width": c["width"],
                "height": c["height"],
                "mask": artifact_url(camera_asset, mask_name),
                "photo": artifact_url(camera_asset, f"{prefix}/reference/images/{name}"),
                "contour": f"/api/comparison/contour/{camera_asset['id']}/{quote(mask_name, safe='/')}",
                "projection": projection.tolist(),
                "view": view.tolist(),
            }
        )
    if not frames:
        raise ValueError("Нет кадров с сохранёнными камерами")
    return {
        "object": object_id,
        "name": figure.name,
        "source_video_sha256": source_hash,
        "frames": frames,
        "candidates": entries,
        "reference_bounds": bounds,
        "unavailable": unavailable,
    }


def gallery_selection(store, identifier):
    path = store.state / "gallery-choices.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    if identifier in saved:
        choices = saved[identifier]
    else:
        choices = []
        project = store.load()
        for figure in project.figures:
            if (
                project.scenes.selected_classes is not None
                and figure.id not in project.scenes.selected_classes
            ):
                continue
            item_id = project.scenes.asset_versions.get(figure.id)
            if not item_id:
                continue
            item = store.artifact(item_id, verify=False)
            meta = item["manifest"]["metadata"]
            choice = meta.get("selection")
            if choice:
                try:
                    data = package_figure(identifier, figure.id)
                except (ValueError, OSError, KeyError):
                    continue
                match = next(
                    (c for c in data["candidates"] if c["mesh_sha256"] == choice["mesh_sha256"]),
                    None,
                )
                if match and data["source_video_sha256"] == choice["source_video_sha256"]:
                    choices.append(
                        {
                            **choice,
                            "id": match["id"],
                            "label": match.get("label", choice.get("label")),
                        }
                    )
    return {"schema_version": 1, "choices": choices, "revision": digest(saved)}
