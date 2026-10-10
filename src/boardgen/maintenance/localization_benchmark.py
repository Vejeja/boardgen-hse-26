from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from boardgen.evaluation.detection_metrics import iou
from boardgen.localization.assets import load_figure
from boardgen.localization.fitting import fit_figure
from boardgen.localization.geometry import blender_camera
from boardgen.localization.registration import prepare_template, register_board
from boardgen.localization.visualization import draw_map, draw_overlay
from boardgen.studio.storage import json_write, sha


def select_frames(records, predictions, confidence=0.5, minimum_iou=0.85):
    by_name = {Path(r["image"]).name: r for r in records}
    selected = []
    for prediction in predictions:
        record = by_name[prediction["image"]]
        if record.get("scene_type") != "board" or not record["labels"]:
            continue
        detections = sorted(
            zip(prediction["confidence"], prediction["classes"], prediction["boxes"], strict=True),
            reverse=True,
        )
        detections = [row for row in detections if row[0] >= confidence]
        if len(detections) != len(record["labels"]):
            continue
        unused = set(range(len(record["labels"])))
        matches = []
        for conf, cls, box in detections:
            choices = [
                (iou(box, record["labels"][index][1:]), index)
                for index in unused
                if cls == record["labels"][index][0]
            ]
            overlap, index = max(choices, default=(0, -1))
            if overlap < minimum_iou:
                break
            unused.remove(index)
            matches.append(
                {
                    "confidence": conf,
                    "class_id": int(cls),
                    "box": box,
                    "truth_index": index,
                    "iou": overlap,
                }
            )
        if not unused:
            selected.append((record, matches))
    return selected


def benchmark(
    dataset: Path, predictions: Path, output: Path, *, minimum_iou=0.85, camera_mode="oracle"
):
    if output.exists():
        raise ValueError("Выберите новую папку результатов")
    data = json.loads((dataset / "dataset.json").read_text())
    selected = select_frames(
        data["records"], json.loads(predictions.read_text()), minimum_iou=minimum_iou
    )
    if not selected:
        raise ValueError("Нет кадров, соответствующих критерию отбора")
    output.mkdir(parents=True)
    meshes, templates = {}, {}
    features, models, boards = {}, {}, {}
    (output / "models").mkdir()
    (output / "boards").mkdir()
    frames = []
    start = time.perf_counter()
    for record, matches in selected:
        scene = record["scene"]
        image = cv2.imread(str(dataset / record["image"]))
        if image is None:
            raise ValueError(f"Не читается изображение {record['image']}")
        camera = blender_camera(scene["camera"], scene["image_size"])
        board_size = [scene["board"]["width_m"], scene["board"]["height_m"]]
        board_path = scene["board"]["image"]
        if board_path not in templates:
            templates[board_path] = cv2.imread(board_path)
        template = templates[board_path]
        if template is None:
            raise ValueError(f"Не читается карта {board_path}")
        if board_path not in boards:
            board_file = f"boards/{sha(Path(board_path))[:16]}.png"
            cv2.imwrite(str(output / board_file), template)
            boards[board_path] = board_file
            features[board_path] = prepare_template(template)
        frame_metadata = {
            "image": record["image"],
            "name": record["name"],
            "source": str(dataset / record["image"]),
            "frame_index": 0,
            "timestamp_s": None,
            "overlay": record["name"] + "-overlay.jpg",
            "board": {
                "file": boards[board_path],
                "width_m": board_size[0],
                "height_m": board_size[1],
            },
        }
        registration = {"method": "oracle_blender_pose"}
        if camera_mode == "template":
            try:
                camera, registration = register_board(
                    image, template, camera.K, (), board_size, features=features[board_path]
                )
            except ValueError as error:
                frames.append(
                    {
                        **frame_metadata,
                        "status": "registration_failed",
                        "error": str(error),
                        "figures": [],
                    }
                )
                cv2.imwrite(str(output / frame_metadata["overlay"]), image)
                print(record["name"], "registration_failed", flush=True)
                continue
        figures = []
        geometries = {}
        for match in matches:
            truth = scene["objects"][match["truth_index"]]
            key = (truth["mesh_path"], truth["height_m"])
            if key not in meshes:
                meshes[key] = load_figure(Path(key[0]), key[1])
            geometry = meshes[key]
            cid = match["class_id"]
            if cid not in models:
                mesh_file = f"models/{cid}.glb"
                shutil.copy2(key[0], output / mesh_file)
                models[cid] = {
                    **geometry.metadata(),
                    "file": mesh_file,
                    "name": data["names"][cid],
                    "sha256": sha(Path(key[0])),
                }
            geometries[match["class_id"]] = geometry
            box = np.asarray(match["box"]) * [
                camera.width,
                camera.height,
                camera.width,
                camera.height,
            ]
            result = fit_figure(camera, geometry, box, board_size)
            true_base = geometry.base_at(truth["xy"], truth["yaw_deg"])
            result.update(
                class_id=match["class_id"],
                box=match["box"],
                confidence=match["confidence"],
                detection_iou=match["iou"],
                truth_xy_m=true_base.tolist(),
                truth_placement_xy_m=truth["xy"],
                error_mm=float(np.linalg.norm(np.asarray(result["xy_m"]) - true_base) * 1000),
                bottom_baseline_error_mm=float(
                    np.linalg.norm(camera.plane_point([box[0], box[1] + box[3] / 2]) - true_base)
                    * 1000
                ),
                center_baseline_error_mm=float(
                    np.linalg.norm(camera.plane_point(box[:2]) - true_base) * 1000
                ),
            )
            figures.append(result)
        name = record["name"]
        cv2.imwrite(
            str(output / f"{name}-overlay.jpg"),
            draw_overlay(image, camera, geometries, figures, board_size),
        )
        cv2.imwrite(str(output / f"{name}-map.jpg"), draw_map(template, figures, board_size))
        frames.append(
            {
                **frame_metadata,
                "map": name + "-map.jpg",
                "status": "processed",
                "camera": camera.to_dict(),
                "registration": registration,
                "board_size_m": board_size,
                "figures": figures,
            }
        )
        print(
            name,
            "median error mm",
            round(float(np.median([f["error_mm"] for f in figures])), 2),
            flush=True,
        )
    figures = [f for frame in frames for f in frame.get("figures", [])]

    def summary(key):
        values = [f[key] for f in figures]
        return (
            {
                "median": float(np.median(values)),
                "p90": float(np.percentile(values, 90)),
                "max": float(np.max(values)),
            }
            if values
            else None
        )

    report = {
        "schema": "boardgen.localization-benchmark.v1",
        "camera_mode": camera_mode,
        "selection": {
            "confidence": 0.5,
            "minimum_iou": minimum_iou,
            "all_objects_correct": True,
            "frames": len(selected),
            "warning": "Selected successful detections, not full dataset accuracy",
        },
        "inputs": {
            "dataset_sha256": sha(dataset / "dataset.json"),
            "predictions_sha256": sha(predictions),
            "meshes": {path: sha(Path(path)) for path, _ in meshes},
        },
        "processed_frames": sum(f["status"] == "processed" for f in frames),
        "objects": len(figures),
        "seconds": time.perf_counter() - start,
        "errors_mm": summary("error_mm"),
        "bottom_baseline_mm": summary("bottom_baseline_error_mm"),
        "center_baseline_mm": summary("center_baseline_error_mm"),
        "frames": frames,
        "base_definition": "Circle of coherent lower horizontal support (area-weighted), lower 2% fallback; compare transformed mesh proxies, not manually measured physical bases",
    }
    json_write(output / "report.json", report)
    json_write(
        output / "scene.json",
        {
            "schema": "boardgen.localized-scene.v1",
            "units": "metres",
            "board": frames[0]["board"],
            "models": models,
            "frames": frames,
            "note": "Selected synthetic benchmark; mesh base proxies and approximate yaw",
        },
    )
    with (output / "frames.jsonl").open("w") as stream:
        for frame in frames:
            stream.write(json.dumps(frame, ensure_ascii=False, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate board localization using saved YOLO predictions."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-iou", type=float, default=0.85)
    parser.add_argument("--camera-mode", choices=["oracle", "template"], default="oracle")
    args = parser.parse_args()
    report = benchmark(
        args.dataset,
        args.predictions,
        args.output,
        minimum_iou=args.minimum_iou,
        camera_mode=args.camera_mode,
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "processed_frames",
                    "objects",
                    "seconds",
                    "errors_mm",
                    "bottom_baseline_mm",
                    "center_baseline_mm",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
