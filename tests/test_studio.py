import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from boardgen.studio.api import create_app
from boardgen.studio.datasets import build, labels, split_groups
from boardgen.studio.jobs import Supervisor, cancel, enqueue
from boardgen.studio.models import Figure, Project
from boardgen.studio.rendering import camera
from boardgen.studio.storage import Store, digest


def test_configuration_is_strict_and_overrides_inherit():
    p = Project(
        figures=[Figure(id="a", name="A", video="a.mp4"), Figure(id="b", name="B", video="b.mp4")]
    )
    p.prepare.overrides["a"] = {"shape": {"mesh_resolution": 384}}
    assert p.prepare.resolved("a").shape.mesh_resolution == 384
    assert p.prepare.resolved("b").shape.mesh_resolution == 256
    assert p.prepare.resolved("a").matte.input_size == 1024
    p.prepare.overrides["a"]["shape"]["typo"] = 4
    with pytest.raises(ValidationError):
        Project.model_validate(p.model_dump())
    with pytest.raises(ValidationError):
        Project.model_validate({"scenes": {"camera": {"distance_m": [2, 1]}}})
    with pytest.raises(ValidationError):
        Project.model_validate({"prepare": {"defaults": {"appearance": {"method": "native"}}}})


def test_clip_and_video_identity_do_not_follow_display_name(tmp_path):
    store = Store(tmp_path)
    p = Project(
        figures=[Figure(id="stable", name="A", video="a.mp4", clip={"start_s": 2, "end_s": 8})]
    )
    store.save(p)
    p.figures[0].name = "Renamed"
    store.save(p)
    assert store.load().figures[0].id == "stable"
    assert store.load().figures[0].clip.start_s == 2
    with pytest.raises(ValidationError):
        Figure(id="bad", name="Bad", video="a.mp4", clip={"start_s": 8, "end_s": 2})


def test_artifact_publication_and_corruption_reject_cache(tmp_path):
    store = Store(tmp_path)
    temporary = tmp_path / "temp"
    temporary.mkdir()
    (temporary / "data.yaml").write_text(yaml.safe_dump({"path": str(temporary)}))
    (temporary / "result.json").write_text(json.dumps({"file": str(temporary / "data.yaml")}))
    item = store.publish(temporary, "dataset", "key", {"path": str(temporary)})
    path = Path(item["directory"])
    assert not temporary.exists()
    assert yaml.safe_load((path / "data.yaml").read_text())["path"] == str(path)
    assert store.cached("key")["id"] == item["id"]
    (path / "data.yaml").write_text("corrupt")
    assert store.cached("key") is None
    with pytest.raises(ValueError, match="повреждён"):
        store.artifact(item["id"])


def test_project_save_prevents_lost_updates_and_external_escape(tmp_path):
    store = Store(tmp_path)
    revision = digest(store.load().model_dump())
    p = store.load()
    p.name = "Changed"
    store.save(p, revision)
    with pytest.raises(ValueError, match="другом окне"):
        store.save(Project(), revision)
    with pytest.raises(ValueError, match="не подключена"):
        store.resolve("../private.txt")
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    store.grant(str(outside))
    assert Store(tmp_path).resolve(str(outside)) == outside


def records(tmp_path):
    result = []
    for i in range(8):
        path = tmp_path / f"{i}.png"
        Image.fromarray(
            np.random.default_rng(i).integers(0, 256, (24, 32, 3), dtype=np.uint8)
        ).save(path)
        result.append(
            {
                "image": str(path),
                "labels": [[0, 0.5, 0.5, 0.4, 0.4]],
                "group": f"capture-{i // 2}",
                "origin": "test",
            }
        )
    return result


def test_grouped_split_is_seeded_and_never_splits_capture(tmp_path):
    rows = records(tmp_path)
    a = split_groups(rows, 0.25, 42)
    assert a == split_groups(list(reversed(rows)), 0.25, 42)
    assert set(a.values()) == {"train", "val"}
    for row in rows:
        assert a[row["group"]] == a[row["group"]]
    with pytest.raises(ValueError, match="две независимые"):
        split_groups([rows[0], rows[1]], 0.2, 42)


def test_dataset_yaml_is_disjoint_and_duplicates_fail(tmp_path):
    store = Store(tmp_path)
    p = Project(figures=[Figure(id="a", name="A", video="a.mp4")])
    output = tmp_path / "dataset"
    output.mkdir()
    rows = records(tmp_path)
    report = build(store, p, output, rows)
    data = yaml.safe_load((output / "data.yaml").read_text())
    assert data["train"] != data["val"]
    splits = {r["split"] for r in report["records"]}
    assert splits == {"train", "val"}
    groups = {}
    for record in report["records"]:
        assert groups.setdefault(record["group"], record["split"]) == record["split"]
    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    with pytest.raises(ValueError, match="Дублирующиеся"):
        build(store, p, duplicate, [*rows, {**rows[0], "group": "other"}])


def test_annotation_requires_confirmation_and_valid_bounds(tmp_path):
    missing = tmp_path / "missing.txt"
    with pytest.raises(ValueError, match="Нет разметки"):
        labels(missing, 1)
    assert labels(missing, 1, allow_empty=True) == []
    missing.write_text("0 .9 .5 .5 .2\n")
    with pytest.raises(ValueError, match="границы"):
        labels(missing, 1)
    missing.write_text("1 .5 .5 .2 .2\n")
    with pytest.raises(ValueError, match="класс"):
        labels(missing, 1)


def test_spherical_camera_is_recorded_and_reproducible():
    import math
    import random

    p = Project()
    p.scenes.camera.jitter_xy_m = 0
    first = camera(p.scenes.camera, random.Random(9))
    assert first == camera(p.scenes.camera, random.Random(9))
    assert math.isclose(sum(v * v for v in first["position_m"]) ** 0.5, first["distance_m"])
    assert 25 <= first["elevation_deg"] <= 45


def test_job_snapshots_are_immutable_and_cancel_queue(tmp_path):
    store = Store(tmp_path)
    p = store.load()
    p.dataset.include_synthetic = False
    store.save(p)
    job = enqueue(store, "install", {})
    p.name = "New"
    store.save(p)
    assert store.job(job["id"])["snapshot"]["name"] == "Boardgen"
    assert cancel(store, job["id"])["status"] == "cancelled"
    assert Store(tmp_path).job(job["id"])["status"] == "cancelled"


def test_cancel_terminates_descendant_process_group(tmp_path):
    store = Store(tmp_path)
    store.save(Project(dataset={"include_synthetic": False}))
    job = enqueue(store, "install", {})
    script = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); time.sleep(60)"
    process = subprocess.Popen(
        [sys.executable, "-c", script, "boardgen.studio.worker", str(store.root), job["id"]],
        start_new_session=True,
    )
    try:
        store.update_job(job["id"], status="running", pid=process.pid)
        cancel(store, job["id"])
        assert process.wait(timeout=5) != 0
        Supervisor(store).tick()
        assert store.job(job["id"])["status"] == "cancelled"
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_api_requires_same_origin_token_and_valid_schema(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        data = client.get("/api/project").json()
        assert client.put("/api/project", json={"project": data["project"]}).status_code == 403
        headers = {"X-Studio-Token": data["token"], "X-Workspace": data["root"]}
        assert (
            client.post(
                "/api/fs/browse",
                json={"path": str(tmp_path)},
                headers={**headers, "Origin": "https://evil.test"},
            ).status_code
            == 403
        )
        assert client.get("/api/project", headers={"Host": "evil.test"}).status_code == 403
        invalid = {**data["project"], "typo": True}
        assert (
            client.put("/api/project", json={"project": invalid}, headers=headers).status_code
            == 422
        )
        assert client.get("/api/artifacts/absent/../../project.yaml").status_code in {404, 422}
        assert (
            client.put(
                "/api/project",
                json={"project": data["project"], "revision": data["revision"]},
                headers=headers,
            ).status_code
            == 200
        )


def test_dead_worker_is_interrupted_instead_of_false_success(tmp_path):
    store = Store(tmp_path)
    store.save(Project(dataset={"include_synthetic": False}))
    job = enqueue(store, "install", {})
    store.update_job(job["id"], status="running", pid=99999999)
    Supervisor(store).tick()
    assert store.job(job["id"])["status"] == "interrupted"


def test_publication_recovers_after_rename_before_database_commit(tmp_path):
    store = Store(tmp_path)
    pending = tmp_path / "pending"
    pending.mkdir()
    (pending / "result.txt").write_text("complete")
    item = store.publish(pending, "figure", "recoverable", {"figure_id": "a"}, "a")
    with store.connect() as connection:
        connection.execute("DELETE FROM artifacts WHERE id=?", (item["id"],))
    restored = Store(tmp_path).cached("recoverable")
    assert restored["id"] == item["id"]
    assert restored["figure_id"] == "a"
    (Path(restored["directory"]) / "unexpected.txt").write_text("mutation")
    with pytest.raises(ValueError, match="состав файлов"):
        store.artifact(restored["id"])


def test_unknown_job_parameters_are_rejected(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(ValidationError):
        enqueue(store, "install", {"epochs": 12})
    with pytest.raises(ValidationError):
        enqueue(store, "evaluate", {"dataset_id": "a", "model_id": "b", "conf": 0.2})


def test_unrelated_recycled_pid_is_not_cancelled(tmp_path):
    store = Store(tmp_path)
    store.save(Project(dataset={"include_synthetic": False}))
    job = enqueue(store, "install", {})
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
    )
    try:
        store.update_job(job["id"], status="running", pid=process.pid)
        cancel(store, job["id"])
        Supervisor(store).tick()
        assert process.poll() is None
        assert store.job(job["id"])["status"] == "cancelled"
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_yolo_label_caches_cannot_mutate_frozen_dataset(tmp_path):
    from boardgen.studio.training import runtime_dataset

    store = Store(tmp_path)
    pending = tmp_path / "pending"
    pending.mkdir()
    (pending / "data.yaml").write_text("path: old\ntrain: images/train\nval: images/val\n")
    artifact = store.publish(pending, "dataset", "data", {})
    working = tmp_path / "ultralytics-working-copy"
    yaml_path = runtime_dataset(artifact, working)
    (working / "labels.cache").write_bytes(b"cache")
    assert yaml.safe_load(yaml_path.read_text())["path"] == str(working)
    assert store.artifact(artifact["id"])["id"] == artifact["id"]
    assert not (Path(artifact["directory"]) / "labels.cache").exists()


def test_new_dataset_version_does_not_hide_evaluation_leakage():
    from boardgen.studio.training import guard_evaluation_split

    training = {
        "records": [
            {"split": "train", "sha256": "a", "group": "capture-a"},
            {"split": "val", "sha256": "b", "group": "capture-b"},
        ]
    }
    guard_evaluation_split(
        training, {"records": [{"split": "val", "sha256": "c", "group": "capture-c"}]}, "val"
    )
    for row in ({"sha256": "a", "group": "other"}, {"sha256": "new", "group": "capture-b"}):
        with pytest.raises(ValueError, match="использованные при обучении"):
            guard_evaluation_split(training, {"records": [{"split": "test", **row}]}, "test")


def test_fixed_yolo_splits_are_retained_without_capture_csv():
    rows = [
        {"group": "a", "fixed_split": "train"},
        {"group": "b", "fixed_split": "val"},
        {"group": "c", "fixed_split": "test"},
    ]
    assert split_groups(rows, 0.9, 6) == {"a": "train", "b": "val"}
    with pytest.raises(ValueError, match="разным разделам"):
        split_groups([*rows, {"group": "a", "fixed_split": "val"}], 0.2, 6)


def test_mv_texture_changes_reuse_geometry(tmp_path):
    from boardgen.studio.preparation import frame_key, geometry_stage

    store = Store(tmp_path)
    (tmp_path / "source.mp4").write_bytes(b"source")
    project = Project(figures=[Figure(id="a", name="A", video="source.mp4")])
    figure = project.figures[0]
    versions = {"code": {}, "packages": {}, "tools": {}, "neural": {}}
    keys = []

    def cached(key):
        keys.append(key)
        return {"id": "existing"}

    store.cached = cached
    first_frame_key = frame_key(store, figure, project.prepare.resolved("a"), versions)
    geometry_stage(store, project, figure, {"cache_key": "frames"}, tmp_path, versions)
    project.prepare.defaults.appearance.texture_size = 1024
    project.prepare.defaults.appearance.photo_views = 4
    assert frame_key(store, figure, project.prepare.resolved("a"), versions) == first_frame_key
    geometry_stage(store, project, figure, {"cache_key": "frames"}, tmp_path, versions)
    assert keys[0] == keys[1]
    project.prepare.defaults.shape.mesh_resolution = 384
    geometry_stage(store, project, figure, {"cache_key": "frames"}, tmp_path, versions)
    assert keys[2] != keys[1]


def test_independent_test_only_source_can_be_published(tmp_path):
    store = Store(tmp_path)
    source = tmp_path / "real"
    (source / "images/test").mkdir(parents=True)
    (source / "labels/test").mkdir(parents=True)
    Image.new("RGB", (24, 32), "red").save(source / "images/test/a.png")
    (source / "labels/test/a.txt").write_text("0 .5 .5 .4 .4\n")
    (source / "data.yaml").write_text(
        "path: .\ntrain: null\nval: null\ntest: images/test\nnames: [A]\n"
    )
    project = Project(
        dataset={
            "include_synthetic": False,
            "real_sources": [{"path": str(source / "data.yaml"), "role": "test_only"}],
        }
    )
    target = tmp_path / "output"
    target.mkdir()
    report = build(store, project, target, [])
    assert report["counts"] == {"test": 1, "train": 0, "val": 0}
    document = yaml.safe_load((target / "data.yaml").read_text())
    assert document["train"] is None and document["val"] is None
    assert document["test"] == "images/test"
    assert report["names"] == ["A"]


def test_checkpoint_interruption_preserves_previous_whole_epoch(tmp_path, monkeypatch):
    from boardgen.studio import training

    source = tmp_path / "last.pt"
    target = tmp_path / "resume-last.pt"
    source.write_bytes(b"new whole epoch")
    target.write_bytes(b"old whole epoch")

    def interrupted_copy(*_):
        target.with_suffix(".pending").write_bytes(b"partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(training.shutil, "copyfile", interrupted_copy)
    with pytest.raises(KeyboardInterrupt):
        training.checkpoint_copy(source, target)
    assert target.read_bytes() == b"old whole epoch"
