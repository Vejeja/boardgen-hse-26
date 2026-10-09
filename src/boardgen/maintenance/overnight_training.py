import argparse
import fcntl
import gc
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from itertools import batched
from pathlib import Path

from boardgen.evaluation.detection_metrics import score_predictions
from boardgen.rendering.blender_render import BlenderSession
from boardgen.studio import backgrounds, datasets, environment, rendering, training
from boardgen.studio.models import Project
from boardgen.studio.stages import selected_assets
from boardgen.studio.storage import Store, digest, json_write, sha

SEED = 10102026
COUNTS = {"train": 1800, "val": 320, "test": 320}
CHUNK = 32


def setup(source, destination):
    if destination.exists():
        raise ValueError("Use a new project directory")
    original = Store(source)
    project = original.load()
    chosen = project.scenes.selected_classes
    if len(chosen or []) != 13:
        raise ValueError("Expected the preserved selection of 13 classes")
    store = Store(destination)
    store.grant(str(environment.REPOSITORY))
    mapping = json.loads((source / "class-map.json").read_text())
    for figure in project.figures:
        asset = original.artifact(project.scenes.asset_versions[figure.id])
        folder = store.state / "import" / figure.id
        shutil.copytree(asset["directory"], folder)
        (folder / "manifest.json").unlink()
        item = store.publish(
            folder, "figure", asset["cache_key"], asset["manifest"]["metadata"], figure.id
        )
        project.scenes.asset_versions[figure.id] = item["id"]
        row = next(r for r in mapping["figures"] if r["object"] == figure.id)
        row["asset_id"] = item["id"]
        row["mesh_sha256"] = item["manifest"]["files"]["figure.glb"]
    root = environment.REPOSITORY

    def board(name, height=0.5):
        return {"image": str(root / "data/board" / name), "width_m": 0.5, "height_m": height}

    development = [
        board(n)
        for n in (
            "board1.png",
            "kroote_grass_river.jpg",
            "kroote_stone_lakes.jpg",
            "kroote_desert_island.jpg",
        )
    ]
    test = [board("board2.jpeg", 0.28095238095238095), board("ferrin_cavern.jpg")]
    payload = project.model_dump(mode="json")
    payload["name"] = "Ночной эксперимент: 13 фигур, разнообразные Blender-сцены"
    payload["scenes"].update(
        count=sum(COUNTS.values()),
        seed=SEED,
        image_size=[1152, 1152],
        objects_per_scene=[2, 6],
        negative_fraction=0.1,
        min_side_px=16,
        max_attempts=20,
        render_samples=16,
        scene_type="mixed",
        balanced_classes=True,
        stratified_negatives=True,
        light_energy=[90, 280],
        exposure_jitter_ev=0.6,
        boards=development,
        skyboxes=[str(root / "data/skybox/room.jpg")],
        camera={
            "distance_m": [0.65, 1.1],
            "elevation_deg": [18, 90],
            "azimuth_deg": [-180, 180],
            "fov_deg": 45,
            "jitter_xy_m": 0.03,
            "jitter_roll_deg": 7,
            "exposure_ev": -1.1,
        },
        platform={
            "radius_m": 0.075,
            "height_m": 0.018,
            "elevation_deg": [10, 85],
            "frame_fraction": [0.22, 0.55],
        },
        split_sources={
            "train": {"boards": development, "skyboxes": [str(root / "data/skybox/room.jpg")]},
            "val": {"boards": development, "skyboxes": [str(root / "data/skybox/room.jpg")]},
            "test": {"boards": test, "skyboxes": [str(root / "data/skybox/room2.jpg")]},
        },
    )
    payload["dataset"].update(
        name="nightly-13",
        seed=SEED,
        val_fraction=320 / 2440,
        test_fraction=320 / 2440,
        real_sources=[],
        near_duplicates="error",
        background_policy="test_only",
    )
    payload["training"].update(
        weights="yolo11s.pt",
        epochs=150,
        time=None,
        imgsz=960,
        batch=8,
        patience=25,
        workers=0,
        optimizer="AdamW",
        lr0=0.001,
        nbs=16,
        warmup_epochs=3,
        weight_decay=0.0005,
        cos_lr=True,
        seed=SEED,
        mosaic=0.10,
        close_mosaic=15,
        hsv_h=0.005,
        hsv_s=0.20,
        hsv_v=0.30,
        fliplr=0,
        flipud=0,
        mixup=0,
        copy_paste=0,
        scale=0.25,
        translate=0.08,
    )
    payload["evaluation"].update(split="val", imgsz=960, confidence=0.45, nms_iou=0.45)
    project = Project.model_validate(payload)
    backgrounds.validate_sources(store, project)
    store.save(project)
    json_write(destination / "class-map.json", mapping)
    json_write(
        destination / "experiment.json",
        {
            "source_project": str(source),
            "classes": chosen,
            "targets": COUNTS,
            "budget_seconds": 4 * 3600,
            "hard_limit_seconds": 5 * 3600,
            "generation_limit_seconds": 75 * 60,
            "evaluation_reserve_seconds": 12 * 60,
            "board_fraction": 0.7,
            "camera_bands": [[18, 40], [40, 65], [65, 85], [85, 90]],
            "camera_band_fractions": [0.20, 0.35, 0.30, 0.15],
            "background_assignment": "New protocol: Kroote family and board1 in development; Ferrin and board2 held out.",
            "prior_results": "Preserved; this experiment starts from official COCO weights.",
            "project": project.model_dump(mode="json"),
            "runner_sha256": sha(Path(__file__)),
        },
    )
    print(f"Prepared {destination}", flush=True)


def chunk_project(project, split, index, count):
    p = project.model_copy(deep=True)
    p.scenes.seed = SEED + {"train": 100000, "val": 200000, "test": 300000}[split] + index * 100003
    p.scenes.count = count
    p.scenes.scene_type = "board" if index % 10 < 7 else "platform"

    block, within = divmod(index, 10)
    kind_index = block * 7 + within if within < 7 else block * 3 + within - 7
    bands = [1, 2, 3, 0, 1, 2, 1, 0, 2, 1, 3, 2, 1, 0, 2, 1, 3, 2, 1, 0]
    if within >= 7:
        bands = [0, 2, 3, 1, 2, 1, 1, 0, 2, 1, 3, 2, 1, 0, 2, 1, 3, 2, 1, 0]
    band = bands[kind_index % len(bands)]
    p.scenes.camera.elevation_deg = [(18, 40), (40, 65), (65, 85), (85, 90)][band]
    p.scenes.platform.elevation_deg = [(10, 35), (35, 55), (55, 75), (75, 90)][band]

    sources = p.scenes.split_sources[split]
    sources.boards = [sources.boards[index % len(sources.boards)]]
    return p


def smoke(store, output):
    p = store.load()
    assets = selected_assets(store, p, {})
    with BlenderSession(environment.blender(), recycle_after=3) as session:
        for index, (kind, angle, negative) in enumerate(
            [
                ("board", 45, False),
                ("board", 90, False),
                ("platform", 30, False),
                ("board", 45, True),
            ]
        ):
            q = p.model_copy(deep=True)
            q.scenes.scene_type = kind
            q.scenes.image_size = (512, 512)
            q.scenes.camera.elevation_deg = (angle, angle)
            q.scenes.negative_fraction = 0.5 if negative else 0
            q.scenes.stratified_negatives = False

            for attempt in range(100):
                q.scenes.seed = SEED + index * 1000 + attempt
                rows = rendering.render(
                    store,
                    q,
                    output / f"smoke-{index}-{attempt}",
                    assets,
                    1,
                    split="train",
                    session=session,
                )
                if bool(rows[0]["labels"]) != negative:
                    break
            else:
                raise RuntimeError("Smoke could not produce its requested scene")
            print(
                f"SMOKE OK: {kind}, elevation={angle}, labels={len(rows[0]['labels'])}", flush=True
            )


def remove_cross_split_duplicates(records):
    kept, removed, buckets, checksums = [], [], defaultdict(set), set()
    for row in records:
        checksum = sha(Path(row["image"]))
        value = datasets.image_hash(row["image"])
        candidates = set()
        for block in range(4):
            candidates.update(buckets[(block, (value >> (block * 16)) & 65535)])
        duplicate = checksum in checksums or any(
            kept[i]["fixed_split"] != row["fixed_split"]
            and (value ^ kept[i]["dedup_hash"]).bit_count() <= 3
            for i in candidates
        )
        if duplicate:
            removed.append({"image": row["image"], "split": row["fixed_split"]})
            continue
        index = len(kept)
        for block in range(4):
            buckets[(block, (value >> (block * 16)) & 65535)].add(index)
        checksums.add(checksum)
        kept.append({**row, "dedup_hash": value})
    return kept, removed


def predict_batches(model, paths, *, size, device, batch_size=4):
    predictions = []
    for batch in batched(paths, batch_size, strict=False):
        for result in model.predict(
            list(batch),
            stream=True,
            batch=batch_size,
            imgsz=size,
            conf=0.10,
            iou=0.45,
            device=device,
            verbose=False,
        ):
            predictions.append(
                {
                    "image": Path(result.path).name,
                    "boxes": result.boxes.xywhn.cpu().tolist(),
                    "classes": result.boxes.cls.cpu().tolist(),
                    "confidence": result.boxes.conf.cpu().tolist(),
                }
            )
    return predictions


def select_operating_point(store, project, dataset, model_id, output, deadline):
    model, _, model_path = training.trained_model(store, model_id)
    records = json.loads((Path(dataset["directory"]) / "dataset.json").read_text())["records"]
    validation = [r for r in records if r["split"] == "val"]
    trials = []
    for size in (960, 768):
        if trials and time.time() > deadline - 5 * 60:
            break
        paths = [str(Path(dataset["directory"]) / r["image"]) for r in validation]
        predictions = predict_batches(model, paths, size=size, device=training.device())
        json_write(output / f"val-predictions-{size}.json", predictions)
        for confidence in (0.20, 0.30, 0.40, 0.45, 0.50, 0.60):
            score = score_predictions(predictions, validation, len(project.figures), confidence)
            score.update(imgsz=size, worst_f1=min(r["f1"] for r in score["per_class"]))
            trials.append(score)
    winner = max(trials, key=lambda r: (r["macro_f1"], r["worst_f1"], -r["imgsz"]))
    json_write(output / "validation-search.json", trials)
    project.evaluation.imgsz = winner["imgsz"]
    project.evaluation.confidence = winner["confidence"]
    project.evaluation.split = "test"
    store.save(project)
    json_write(
        output / "decision.json",
        {
            "model_id": model_id,
            "dataset_id": dataset["id"],
            "selected_on": "val",
            "model_sha256": sha(model_path),
            "confidence": winner["confidence"],
            "imgsz": winner["imgsz"],
            "nms_iou": 0.45,
            "macro_f1_val": winner["macro_f1"],
            "frozen_at_unix": time.time(),
        },
    )


def resume_evaluation(store, output):
    previous = json.loads((output / "progress.json").read_text())
    if previous["stage"] == "complete":
        print("Evaluation already complete", flush=True)
        return
    model_id = json.loads((output / "published-model.json").read_text())["id"]
    model = store.artifact(model_id)
    dataset = store.artifact(model["manifest"]["metadata"]["dataset_id"])
    project = store.load()
    recovery_started = time.time()
    recovery = output / f"recovery-{time.time_ns()}"
    recovery.mkdir()
    json_write(recovery / "previous-progress.json", previous)
    shutil.copy2(Path(__file__), recovery / "runner.py")

    def progress(stage, **values):
        json_write(
            output / "progress.json",
            {
                "stage": stage,
                "started_unix": previous["started_unix"],
                "updated_unix": time.time(),
                "elapsed_seconds": time.time() - previous["started_unix"],
                "recovery_started_unix": recovery_started,
                "recovery_directory": str(recovery),
                "recovery_elapsed_seconds": time.time() - recovery_started,
                "model_id": model_id,
                "dataset_id": dataset["id"],
                **values,
            },
        )

    try:
        with (environment.REPOSITORY / ".boardgen/compute.lock").open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            versions = environment.fingerprint()
            progress("validation")
            select_operating_point(
                store, project, dataset, model_id, recovery, time.time() + 30 * 60
            )
            for name in ("decision.json", "validation-search.json"):
                shutil.copy2(recovery / name, output / name)
            gc.collect()
            if training.device() == "mps":
                import torch

                torch.mps.empty_cache()
            progress("test")
            evaluation_project = project.model_copy(deep=True)
            evaluation_project.training.batch = 4
            result = training.evaluate(
                store,
                evaluation_project,
                {"dataset_id": dataset["id"], "model_id": model_id},
                recovery,
                versions,
            )
        evaluation = store.artifact(result["artifacts"][0])
        json_write(output / "results.json", evaluation["manifest"]["metadata"])
        progress(
            "complete",
            evaluation_id=evaluation["id"],
            counts=dataset["manifest"]["metadata"]["counts"],
            metrics=evaluation["manifest"]["metadata"]["metrics"],
        )
        print(
            json.dumps(evaluation["manifest"]["metadata"]["operating_point"], indent=2), flush=True
        )
    except BaseException as error:
        progress("failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise


def run(store, output):
    started = time.time()
    deadline = started + 4 * 3600
    finished = threading.Event()

    def progress(stage, **values):
        json_write(
            output / "progress.json",
            {
                "stage": stage,
                "started_unix": started,
                "updated_unix": time.time(),
                "deadline_unix": deadline,
                "elapsed_seconds": time.time() - started,
                **values,
            },
        )

    def watchdog():
        if finished.wait(5 * 3600 - 30):
            return
        progress("hard_time_limit")
        for job in store.jobs(statuses=("running",)):
            if job["pid"]:
                os.killpg(job["pid"], signal.SIGTERM)
        os.killpg(os.getpgrp(), signal.SIGTERM)

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        project = store.load()
        if (output / "progress.json").exists():
            raise ValueError("This experiment has already started")
        progress("verifying_inputs")
        assets = selected_assets(store, project, {})
        backgrounds.validate_sources(store, project)
        weights = environment.TOOLS / "yolo/yolo11s.pt"
        if not weights.is_file():
            raise ValueError("Official YOLO11s weights must be available before rendering")
        versions = environment.fingerprint()
        json_write(
            output / "runtime.json", {"versions": versions, "initial_weights_sha256": sha(weights)}
        )
        records = []
        counts = Counter()
        generation_deadline = started + 75 * 60
        with (environment.REPOSITORY / ".boardgen/compute.lock").open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            with BlenderSession(environment.blender()) as session:
                for split in ("val", "test", "train"):
                    for index in range(math.ceil(COUNTS[split] / CHUNK)):
                        if (
                            split == "train"
                            and counts[split] >= 520
                            and time.time() > generation_deadline
                        ):
                            break
                        count = min(CHUNK, COUNTS[split] - counts[split])
                        p = chunk_project(project, split, index, count)
                        folder = output / "renders" / split / f"chunk-{index:03d}"
                        for attempt in range(3):
                            try:
                                batch = rendering.render(
                                    store,
                                    p,
                                    folder / f"attempt-{attempt}",
                                    assets,
                                    count,
                                    split=split,
                                    session=session,
                                )
                                break
                            except (RuntimeError, subprocess.SubprocessError, ValueError) as error:
                                print(f"Render chunk retry {attempt + 1}: {error}", flush=True)
                                session.close()
                                p.scenes.seed += 7000001
                        else:
                            raise RuntimeError(f"Rendering failed: {split}, chunk {index}")
                        records.extend(batch)
                        counts[split] += len(batch)
                        json_write(output / "render-records.json", records)
                        progress("generating", counts=dict(counts), current_split=split)
            records, removed = remove_cross_split_duplicates(records)
            json_write(output / "duplicate-filter.json", removed)
            support = {
                s: Counter(int(b[0]) for r in records if r["fixed_split"] == s for b in r["labels"])
                for s in COUNTS
            }
            if any(support[s][c] < (30 if s == "train" else 15) for s in COUNTS for c in range(13)):
                raise ValueError(f"Insufficient per-class support: {support}")
            temporary = store.state / "import/nightly-dataset"
            temporary.mkdir()
            report = datasets.build(store, project, temporary, records)
            meta = {k: v for k, v in report.items() if k != "records"}
            meta.update(project=project.model_dump(mode="json"), versions=versions)
            dataset = store.publish(temporary, "dataset", digest(meta), meta)
        json_write(
            output / "dataset-result.json",
            {
                "id": dataset["id"],
                "directory": dataset["directory"],
                "counts": report["counts"],
                "support": support,
                "generation_seconds": time.time() - started,
                "removed_duplicates": len(removed),
            },
        )
        project.scenes.count = sum(report["counts"].values())
        project.training.time = max(0.05, (deadline - time.time() - 12 * 60) / 3600)
        store.save(project)
        params = {"dataset_id": dataset["id"], "deadline_unix": deadline - 12 * 60}
        json_write(output / "train-parameters.json", params)
        progress("training", counts=report["counts"], training_hours=project.training.time)
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "boardgen.studio.cli",
                "--project",
                str(store.root),
                "run",
                "train",
                "--parameters",
                str(output / "train-parameters.json"),
            ],
            check=True,
        )
        job = next(j for j in store.jobs() if j["kind"] == "train")
        model_id = job["result"]["artifacts"][0]
        json_write(
            output / "published-model.json", store.artifact(model_id, verify=False)["manifest"]
        )
        progress("validation", model_id=model_id, dataset_id=dataset["id"])
        with (environment.REPOSITORY / ".boardgen/compute.lock").open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX)
            select_operating_point(store, project, dataset, model_id, output, deadline)
            progress("test", model_id=model_id)
            result = training.evaluate(
                store,
                project,
                {"dataset_id": dataset["id"], "model_id": model_id},
                output,
                versions,
            )
        evaluation = store.artifact(result["artifacts"][0], verify=False)
        json_write(output / "results.json", evaluation["manifest"]["metadata"])
        progress(
            "complete",
            model_id=model_id,
            dataset_id=dataset["id"],
            evaluation_id=evaluation["id"],
            counts=report["counts"],
            metrics=evaluation["manifest"]["metadata"]["metrics"],
        )
    except BaseException as error:
        progress("failed", error=f"{type(error).__name__}: {error}")
        traceback.print_exc()
        raise
    finally:
        finished.set()


def main():
    parser = argparse.ArgumentParser(
        description="Blender dataset generation and YOLO training for 13 classes."
    )
    parser.add_argument("action", choices=("setup", "smoke", "launch", "run", "finish"))
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    if args.action == "setup" and args.source is None:
        parser.error("setup requires --source")
    destination = args.project.resolve()
    output = environment.REPOSITORY / "out" / destination.name
    output.mkdir(parents=True, exist_ok=True)
    if args.action == "setup":
        setup(args.source.resolve(), destination)
    elif args.action == "smoke":
        smoke(Store(destination), output)
    elif args.action == "finish":
        resume_evaluation(Store(destination), output)
    elif args.action == "launch":
        if (output / "launch.json").exists() or (output / "progress.json").exists():
            raise ValueError("This experiment has already been launched")
        shutil.copy2(Path(__file__), output / "runner.py")
        with (output / "run.log").open("a") as log:
            process = subprocess.Popen(
                [
                    "/usr/bin/caffeinate",
                    "-ims",
                    sys.executable,
                    "-u",
                    str(Path(__file__).resolve()),
                    "run",
                    "--project",
                    str(destination),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                cwd=environment.REPOSITORY,
            )
        launch = {
            "pid": process.pid,
            "project": str(destination),
            "log": str(output / "run.log"),
            "launched_unix": time.time(),
            "runner_sha256": sha(Path(__file__)),
        }
        json_write(output / "launch.json", launch)
        print(json.dumps(launch, indent=2), flush=True)
    else:
        run(Store(destination), output)


if __name__ == "__main__":
    main()
