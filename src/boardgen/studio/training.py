from __future__ import annotations

import csv
import json
import os
import shutil
import time
from pathlib import Path

from . import environment
from .storage import digest, json_write, sha

os.environ.setdefault("YOLO_CONFIG_DIR", str(environment.REPOSITORY / ".boardgen/ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")

KNOWN_WEIGHTS = {f"yolo{version}{size}.pt" for version in ("11", "26") for size in "nsmlx"}


def device():
    import torch

    return (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def load_weights(store, value):
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    if value in KNOWN_WEIGHTS:
        folder = environment.TOOLS / "yolo"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / value

        model = YOLO(str(path))
        if not path.is_file():
            raise ValueError("Веса YOLO не загрузились")
    else:
        path = store.resolve(value, exists=True)
        if path.suffix != ".pt":
            raise ValueError("Выберите веса YOLO detect в формате .pt")
        model = YOLO(str(path))
    if model.task != "detect":
        raise ValueError("Проект поддерживает YOLO detect; выберите модель обнаружения объектов")
    return model, path


def dataset(store, artifact_id):
    result = store.artifact(artifact_id)
    if result["kind"] != "dataset":
        raise ValueError("Выберите опубликованную версию датасета")
    return result, Path(result["directory"]) / "data.yaml"


def runtime_dataset(artifact, destination):
    import yaml

    source = Path(artifact["directory"])
    destination.mkdir(parents=True)
    for name in ("images", "labels"):
        if (source / name).is_dir():
            shutil.copytree(source / name, destination / name)
    path = destination / "data.yaml"
    document = yaml.safe_load((source / "data.yaml").read_text())
    document["path"] = str(destination)
    path.write_text(yaml.safe_dump(document, sort_keys=False, allow_unicode=True))
    return path


def checkpoint_copy(source, destination):
    partial = destination.with_suffix(".pending")
    shutil.copyfile(source, partial)
    with partial.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(partial, destination)


def train(store, project, parameters, folder, versions):
    data, yaml_path = dataset(store, parameters["dataset_id"])
    if not all(data["manifest"]["metadata"]["counts"].get(k) for k in ("train", "val")):
        raise ValueError(
            "Для обучения нужны непустые train и val; тестовая выборка используется только для оценки"
        )
    if "scalar" in data["manifest"]["metadata"].get("synthetic_annotation_encodings", ["scalar"]):
        report = json.loads((Path(data["directory"]) / "dataset.json").read_text())
        if any(r.get("origin") == "synthetic" for r in report["records"]):
            raise ValueError(
                "Соберите датасет заново: прежние скалярные ID-маски могут искажать рамки на сглаженных краях"
            )
    if data["manifest"]["metadata"].get("near_duplicate_count") and not parameters.get(
        "accept_near_duplicates"
    ):
        raise ValueError("Просмотрите похожие пары в датасете и подтвердите их перед обучением")
    model, weights = load_weights(store, project.training.weights)
    selected_device = device()
    temporary = folder / "training"
    temporary.mkdir()
    yaml_path = runtime_dataset(data, temporary / "input-dataset")
    args = {
        **project.training.model_dump(exclude={"weights"}),
        "data": str(yaml_path),
        "device": selected_device,
        "project": str(temporary),
        "name": "run",
        "exist_ok": False,
        "plots": True,
        "amp": selected_device != "mps",
    }
    resume_id = parameters.get("resume_job_id")
    if resume_id:
        from ultralytics import YOLO

        previous = store.state / "jobs" / resume_id / "training"
        shutil.copytree(previous / "run", temporary / "run")
        checkpoint = temporary / "resume-last.pt"
        shutil.copy2(previous / "resume-last.pt", checkpoint)
        if (previous / "resume-best.pt").exists():
            shutil.copy2(previous / "resume-best.pt", temporary / "run/weights/best.pt")
        model = YOLO(str(checkpoint))
        args = {
            "resume": str(checkpoint),
            "data": str(yaml_path),
            "save_dir": str(temporary / "run"),
            "device": selected_device,
        }
        print(f"Продолжение сохранённой эпохи запуска {resume_id}", flush=True)
    json_write(
        temporary / "resume-state.json",
        {
            "dataset_id": data["id"],
            "dataset_manifest_sha256": sha(Path(data["directory"]) / "manifest.json"),
            "training": project.training.model_dump(),
            "arguments": args,
        },
    )
    print(f"YOLO: {selected_device}, датасет {data['id']}", flush=True)

    def save_epoch(trainer):
        checkpoint_copy(trainer.last, temporary / "resume-last.pt")
        if trainer.best.exists():
            checkpoint_copy(trainer.best, temporary / "resume-best.pt")

    model.add_callback("on_model_save", save_epoch)
    deadline = parameters.get("deadline_unix")
    if deadline:

        def stop_at_deadline(trainer):
            if time.time() >= deadline:
                trainer.stop = True
                print("Лимит времени: сохраняется завершённая эпоха", flush=True)

        model.add_callback("on_train_epoch_end", stop_at_deadline)
    model.train(**args)
    best = temporary / "run/weights/best.pt"
    last = temporary / "run/weights/last.pt"
    if not best.is_file() or not last.is_file():
        raise ValueError("Обучение не создало best.pt и last.pt")
    rows = []
    with (temporary / "run/results.csv").open() as stream:
        rows = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(stream)]
    meta = {
        "dataset_id": data["id"],
        "dataset_counts": data["manifest"]["metadata"]["counts"],
        "class_names": data["manifest"]["metadata"]["names"],
        "dataset_manifest_sha256": sha(Path(data["directory"]) / "manifest.json"),
        "weights": str(weights),
        "weights_sha256": sha(weights),
        "best": "run/weights/best.pt",
        "last": "run/weights/last.pt",
        "device": selected_device,
        "arguments": args,
        "history": rows,
        "versions": versions,
        "near_duplicates_reviewed": bool(parameters.get("accept_near_duplicates")),
        "resumed_from_job": resume_id,
    }
    meta["effective_arguments"] = vars(model.trainer.args)
    for name in ("resume-last.pt", "resume-best.pt", "resume-last.pending", "resume-best.pending"):
        (temporary / name).unlink(missing_ok=True)
    shutil.rmtree(temporary / "input-dataset")
    json_write(temporary / "training.json", meta)
    result = store.publish(temporary, "train", digest(meta), meta)
    return {"artifacts": [result["id"]]}


def trained_model(store, artifact_id):
    from ultralytics import YOLO

    result = store.artifact(artifact_id)
    if result["kind"] != "train":
        raise ValueError("Выберите результат обучения")
    path = Path(result["directory"]) / result["manifest"]["metadata"]["best"]
    return YOLO(str(path)), result, path


def guard_evaluation_split(training_dataset, evaluation_dataset, split):
    seen_splits = {"train", "val"} if split == "test" else {"train"}
    seen = [r for r in training_dataset["records"] if r["split"] in seen_splits]
    target = [r for r in evaluation_dataset["records"] if r["split"] == split]
    hashes = {r["sha256"] for r in seen}
    groups = {r["group"] for r in seen}
    if any(r["sha256"] in hashes or r["group"] in groups for r in target):
        raise ValueError(
            "Выборка для оценки содержит изображения или группы, использованные при обучении"
        )
    policy = evaluation_dataset.get("background_audit", {}).get("policy", "none")
    if policy != "none":
        from .datasets import audit_backgrounds

        audit_backgrounds([*seen, *target], policy)


def inference_size(project, training):
    return project.evaluation.imgsz or training["manifest"]["metadata"].get(
        "effective_arguments", {}
    ).get("imgsz", 640)


def evaluate(store, project, parameters, folder, versions):
    data, yaml_path = dataset(store, parameters["dataset_id"])
    meta_data = data["manifest"]["metadata"]
    if not meta_data["counts"].get(project.evaluation.split):
        raise ValueError(
            "В датасете нет выбранного раздела. Настройте долю test для Blender-сцен или подключите отдельный размеченный тест."
        )
    model, training, path = trained_model(store, parameters["model_id"])
    expected_names = training["manifest"]["metadata"].get("dataset_id")
    source_dataset = store.artifact(expected_names)
    if source_dataset["manifest"]["metadata"]["names"] != meta_data["names"]:
        raise ValueError("Порядок классов модели и датасета различается")
    report = json.loads((Path(data["directory"]) / "dataset.json").read_text())
    source_report = json.loads((Path(source_dataset["directory"]) / "dataset.json").read_text())
    guard_evaluation_split(source_report, report, project.evaluation.split)
    temporary = folder / "evaluation"
    temporary.mkdir()
    yaml_path = runtime_dataset(data, temporary / "input-dataset")

    args = {
        "data": str(yaml_path),
        "split": project.evaluation.split,
        "imgsz": inference_size(project, training),
        "batch": project.training.batch,
        "workers": project.training.workers,
        "device": device(),
        "conf": 0.001,
        "iou": project.evaluation.nms_iou,
        "project": str(temporary),
        "name": "run",
        "plots": True,
        "save_json": False,
    }
    metrics = model.val(**args)
    per_class = [
        {
            "id": int(i),
            "name": meta_data["names"][int(i)],
            "ap50": float(metrics.box.ap50[j]),
            "ap50_95": float(metrics.box.ap[j]),
        }
        for j, i in enumerate(metrics.box.ap_class_index)
    ]
    meta = {
        "dataset_id": data["id"],
        "training_id": training["id"],
        "model_sha256": sha(path),
        "split": project.evaluation.split,
        "metrics": {k: float(v) for k, v in metrics.results_dict.items()},
        "per_class": per_class,
        "ap_confidence_cutoff": 0.001,
        "operating_confidence": project.evaluation.confidence,
        "matching_iou": project.evaluation.matching_iou,
        "nms_iou": project.evaluation.nms_iou,
        "arguments": args,
        "versions": versions,
    }

    from boardgen.evaluation.detection_metrics import score_predictions

    records = [r for r in report["records"] if r["split"] == project.evaluation.split]
    predictions = []
    rendered = {}
    for record in records:
        prediction = model.predict(
            str(Path(data["directory"]) / record["image"]),
            conf=project.evaluation.confidence,
            iou=project.evaluation.nms_iou,
            device=args["device"],
            imgsz=args["imgsz"],
            verbose=False,
        )[0]
        name = Path(record["image"]).name
        predictions.append(
            {
                "image": name,
                "boxes": prediction.boxes.xywhn.cpu().tolist(),
                "classes": prediction.boxes.cls.cpu().tolist(),
                "confidence": prediction.boxes.conf.cpu().tolist(),
            }
        )
        rendered[name] = prediction
    scores = score_predictions(
        predictions,
        records,
        len(meta_data["names"]),
        project.evaluation.confidence,
        project.evaluation.matching_iou,
    )
    meta["operating_point"] = scores
    meta["matching_rule"] = "confidence_ordered_same_class_best_unused_ground_truth"
    json_write(temporary / "predictions.json", predictions)
    by_name = {Path(r["image"]).name: r for r in records}
    pred_by_name = {r["image"]: r for r in predictions}
    errors = []
    for error in scores["errors"]:
        record = by_name[error["image"]]
        prediction = pred_by_name[error["image"]]
        name = record["name"] + ".jpg"
        rendered[error["image"]].save(filename=str(temporary / name))
        errors.append(
            {
                "image": name,
                "source_image": record["image"],
                "false_positives": error["fp"],
                "false_negatives": error["fn"],
                "ground_truth": record["labels"],
                "predictions": [
                    [int(c), *b]
                    for c, b in zip(prediction["classes"], prediction["boxes"], strict=True)
                ],
            }
        )
    meta["errors"] = errors
    shutil.rmtree(temporary / "input-dataset")
    json_write(temporary / "evaluation.json", meta)
    result = store.publish(temporary, "evaluate", digest(meta), meta)
    return {"artifacts": [result["id"]]}


def predict(store, project, parameters, folder, versions):
    model, training, path = trained_model(store, parameters["model_id"])
    source = store.resolve(parameters["source"], exists=True)
    temporary = folder / "prediction"
    temporary.mkdir()
    records = []
    for result in model.predict(
        source=str(source),
        conf=project.evaluation.confidence,
        iou=project.evaluation.nms_iou,
        imgsz=inference_size(project, training),
        device=device(),
        stream=True,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(temporary),
        name="run",
        verbose=False,
    ):
        records.append(
            {
                "path": result.path,
                "boxes": result.boxes.xywhn.cpu().tolist(),
                "classes": result.boxes.cls.cpu().tolist(),
                "confidence": result.boxes.conf.cpu().tolist(),
            }
        )
    meta = {
        "training_id": training["id"],
        "source": str(source),
        "model_sha256": sha(path),
        "confidence": project.evaluation.confidence,
        "nms_iou": project.evaluation.nms_iou,
        "frames": len(records),
        "detection_count": sum(len(r["boxes"]) for r in records),
        "class_names": [model.names[i] for i in sorted(model.names)],
        "imgsz": inference_size(project, training),
        "versions": versions,
    }

    import subprocess

    for video in sorted((temporary / "run").rglob("*")):
        if video.suffix.lower() in {".avi", ".mp4", ".mov", ".mkv"}:
            output = video.with_name(video.stem + "-preview.mp4")
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(video),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output),
                ],
                check=True,
            )
    meta["source_files"] = {
        str(p.relative_to(source)) if source.is_dir() else p.name: sha(p)
        for p in (sorted(source.rglob("*")) if source.is_dir() else [source])
        if p.is_file()
    }
    json_write(temporary / "predictions.json", records)
    result = store.publish(temporary, "predict", digest(meta), meta)
    return {"artifacts": [result["id"]]}
