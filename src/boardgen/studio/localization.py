from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2

from boardgen.localization.assets import load_figure
from boardgen.localization.pipeline import IMAGE_SUFFIXES, VIDEO_SUFFIXES, Localizer, read_frames
from boardgen.localization.visualization import draw_map, draw_overlay

from .storage import digest, json_write, sha


def frozen_models(store, model_id):
    model = store.artifact(model_id, verify=False)
    if model["kind"] != "train":
        raise ValueError("Выберите опубликованную модель обучения")
    dataset = store.artifact(model["manifest"]["metadata"]["dataset_id"], verify=False)
    meta = dataset["manifest"]["metadata"]
    snapshot = meta.get("project", {})
    names = meta.get("names", [])
    figures = {f["name"]: f for f in snapshot.get("figures", [])}
    selections = snapshot.get("scenes", {}).get("asset_versions", {})
    result = {}
    for cid, name in enumerate(names):
        figure = figures.get(name)
        if not figure or figure["id"] not in selections:
            raise ValueError(f"{name}: датасет обучения не содержит выбранную GLB")
        item = store.artifact(selections[figure["id"]], verify=False)
        if item["kind"] != "figure" or item["figure_id"] != figure["id"]:
            raise ValueError(f"{name}: версия GLB не соответствует фигуре")
        result[cid] = {"artifact": item, "figure_id": figure["id"], "name": name}
    if not result:
        raise ValueError("В датасете обучения нет классов")
    return model, dataset, result


def validate_inputs(store, project, parameters):
    source = store.resolve(parameters["source"], exists=True)
    settings = project.localization
    if not settings.board.image:
        raise ValueError("Выберите исходное изображение карты и задайте её размеры")
    store.resolve(settings.board.image, exists=True)
    if not source.is_dir() and source.suffix.lower() not in IMAGE_SUFFIXES | VIDEO_SUFFIXES:
        raise ValueError("Выберите изображение, видео или папку изображений")
    if settings.registration == "corners":
        if len(settings.corners) != 4:
            raise ValueError("Укажите четыре угла поля на первом кадре")
        if source.is_dir() or (
            source.suffix.lower() in VIDEO_SUFFIXES and not settings.fixed_camera
        ):
            raise ValueError("Ручные углы подходят для одного фото или видео с неподвижной камерой")
    return frozen_models(store, parameters["model_id"])


def localize(store, project, parameters, folder, versions):
    from .training import device, inference_size, trained_model

    _, dataset, selected = validate_inputs(store, project, parameters)
    model, training, weights = trained_model(store, parameters["model_id"])
    if [model.names[i] for i in sorted(model.names)] != dataset["manifest"]["metadata"]["names"]:
        raise ValueError("Классы весов не соответствуют датасету и выбранным GLB")
    settings = project.localization
    source = store.resolve(parameters["source"], exists=True)
    board_path = store.resolve(settings.board.image, exists=True)
    template = cv2.imread(str(board_path))
    if template is None:
        raise ValueError("Не читается изображение карты")
    temporary = folder / "localization"
    temporary.mkdir()
    (temporary / "models").mkdir()
    cv2.imwrite(str(temporary / "board.png"), template)
    records = json.loads((Path(dataset["directory"]) / "dataset.json").read_text())["records"]
    heights = {}
    for record in records:
        for obj in record.get("scene", {}).get("objects", []):
            heights.setdefault(obj["class_id"], obj["height_m"])
    geometries, models = {}, {}
    for cid, row in selected.items():
        asset = store.artifact(row["artifact"]["id"])
        relative_mesh = asset["manifest"]["metadata"]["mesh"]
        path = (Path(asset["directory"]) / relative_mesh).resolve()
        if (
            relative_mesh not in asset["manifest"]["files"]
            or not path.is_relative_to(Path(asset["directory"]).resolve())
            or path.suffix.lower() != ".glb"
        ):
            raise ValueError(f"{row['name']}: GLB должна входить в выбранный артефакт")
        override = settings.base_overrides.get(row["figure_id"])
        height = override.height_m if override and override.height_m else heights.get(cid)
        if height is None:
            raise ValueError(
                f"{row['name']}: задайте измеренную высоту в localization.base_overrides"
            )
        geometry = load_figure(
            path,
            height,
            base_xy_m=override.center_xy_m if override else None,
            radius_m=override.radius_m if override else None,
        )
        geometries[cid] = geometry
        relative = f"models/{cid}.glb"
        shutil.copy2(path, temporary / relative)
        models[cid] = {
            **geometry.metadata(),
            "file": relative,
            "name": row["name"],
            "figure_id": row["figure_id"],
            "asset_id": asset["id"],
            "sha256": sha(path),
        }
    localizer = Localizer(settings, template, geometries)
    frames = []
    counts = {"frames": 0, "processed": 0, "registration_failed": 0, "estimated": 0, "uncertain": 0}
    inputs = {}
    preview_count = 0
    with (temporary / "frames.jsonl").open("w") as stream:
        for frame in read_frames(
            source, settings.video_stride, lambda p: store.resolve(p, exists=True)
        ):
            if str(frame.source) not in inputs:
                inputs[str(frame.source)] = sha(frame.source)
            prediction = model.predict(
                frame.image,
                conf=project.evaluation.confidence,
                iou=project.evaluation.nms_iou,
                imgsz=inference_size(project, training),
                device=device(),
                verbose=False,
            )[0]
            detections = [
                {"class_id": int(cid), "confidence": float(conf), "box": box}
                for cid, conf, box in zip(
                    prediction.boxes.cls.cpu().tolist(),
                    prediction.boxes.conf.cpu().tolist(),
                    prediction.boxes.xywhn.cpu().tolist(),
                    strict=True,
                )
            ]
            result, camera = localizer.process(frame, detections)
            counts["frames"] += 1
            counts["processed" if camera else "registration_failed"] += 1
            for item in result["figures"]:
                if item["status"] in {"estimated", "uncertain"}:
                    counts[item["status"]] += 1
            if preview_count < settings.preview_limit:
                name = f"frame_{counts['frames']:06d}"
                result["overlay"] = name + "-overlay.jpg"
                overlay = (
                    draw_overlay(
                        frame.image,
                        camera,
                        geometries,
                        result["figures"],
                        [settings.board.width_m, settings.board.height_m],
                    )
                    if camera
                    else frame.image
                )
                cv2.imwrite(str(temporary / result["overlay"]), overlay)
                if camera:
                    result["map"] = name + "-map.jpg"
                    cv2.imwrite(
                        str(temporary / result["map"]),
                        draw_map(
                            template,
                            result["figures"],
                            [settings.board.width_m, settings.board.height_m],
                        ),
                    )
                frames.append(result)
                preview_count += 1
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            print(
                f"Кадр {frame.index}: {result['status']} · фигур {len(result['figures'])}",
                flush=True,
            )
    if not counts["frames"]:
        raise ValueError("В источнике нет поддерживаемых кадров")
    scene = {
        "schema": "boardgen.localized-scene.v1",
        "convention": "world:x-right-y-map-top-z-up",
        "units": "metres",
        "board": {
            "file": "board.png",
            "width_m": settings.board.width_m,
            "height_m": settings.board.height_m,
        },
        "models": models,
        "frames": frames,
        "note": "GLB placement and base circle are estimates. Yaw is a bbox-fit hypothesis, not observed orientation.",
    }
    json_write(temporary / "scene.json", scene)
    metadata = {
        **counts,
        "training_id": training["id"],
        "dataset_id": dataset["id"],
        "model_sha256": sha(weights),
        "board_sha256": sha(board_path),
        "source_files": inputs,
        "settings": settings.model_dump(mode="json"),
        "scene": "scene.json",
        "coordinates": "frames.jsonl",
        "preview_frames": len(frames),
        "versions": versions,
        "class_names": [row["name"] for row in selected.values()],
    }
    item = store.publish(temporary, "localize", digest(metadata), metadata)
    return {"artifacts": [item["id"]]}
