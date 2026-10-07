import json

import pytest
import yaml
from fastapi.testclient import TestClient

from boardgen.studio.api import create_app
from boardgen.studio.jobs import Supervisor, enqueue
from boardgen.studio.models import Project
from boardgen.studio.sources import connect_videos
from boardgen.studio.storage import Store


def test_new_project_can_export_defaults_and_then_current_settings(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        initial = client.get("/api/project").json()
        exported = client.get("/api/config")
        assert exported.status_code == 200
        assert "project.yaml" in exported.headers["content-disposition"]
        assert Project.model_validate(yaml.safe_load(exported.text)) == Project()
        initial["project"]["training"]["epochs"] = 73
        client.put(
            "/api/project",
            json={"project": initial["project"], "revision": initial["revision"]},
            headers={"X-Studio-Token": initial["token"]},
        ).raise_for_status()
        assert yaml.safe_load(client.get("/api/config").text)["training"]["epochs"] == 73


def test_replace_prunes_figure_references_but_preserves_history_and_common_settings(tmp_path):
    store = Store(tmp_path)
    first = tmp_path / "first"
    first.mkdir()
    (first / "a.mp4").write_bytes(b"source-a")
    old = connect_videos(store, first, "replace")
    identifier = old.figures[0].id
    old.prepare.overrides[identifier] = {"matte": {"method": "classic"}}
    old.scenes.asset_versions[identifier] = "existing-model"
    old.scenes.height_overrides[identifier] = 0.08
    old.scenes.selected_classes = [identifier]
    old.scenes.reference_class = identifier
    old.scenes.scale_basis = "shared_video"
    old.training.epochs = 73
    store.save(old)
    preserved = store.state / "saved-result.txt"
    preserved.write_text("keep")
    second = tmp_path / "second"
    second.mkdir()
    (second / "b.mp4").write_bytes(b"source-b")
    added = connect_videos(store, second, "add")
    assert len(added.figures) == 2 and added.figures[0].id == identifier
    assert added.prepare.overrides[identifier]["matte"]["method"] == "classic"
    replaced = connect_videos(store, second, "replace")
    assert len(replaced.figures) == 1 and replaced.figures[0].name == "b"
    assert (
        replaced.prepare.overrides
        == replaced.scenes.asset_versions
        == replaced.scenes.height_overrides
        == {}
    )
    assert replaced.scenes.selected_classes is None
    assert replaced.scenes.reference_class is None
    assert replaced.scenes.scale_basis == "nominal"
    assert replaced.training.epochs == 73
    assert preserved.read_text() == "keep" and (first / "a.mp4").exists()


def test_reconnecting_same_inputs_keeps_review_and_empty_folder_does_not_reset(tmp_path):
    store = Store(tmp_path)
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "a.mp4").write_bytes(b"source")
    p = connect_videos(store, videos, "replace")
    p.figures[0].name = "Renamed"
    p.figures[0].clip.start_s = 1
    p.figures[0].reviewed_views = {"front": 2}
    store.save(p)
    same = connect_videos(store, videos, "replace")
    assert same.figures == p.figures
    assert len(connect_videos(store, videos, "add").figures) == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="не изменён"):
        connect_videos(store, empty, "replace")
    assert store.load() == p

    (videos / "Renamed.mp4").write_bytes(b"another source")
    merged = connect_videos(store, videos, "replace")
    assert len({f.name for f in merged.figures}) == 2
    assert next(f for f in merged.figures if f.id == p.figures[0].id) == p.figures[0]




def test_artifact_list_omits_heavy_payload_but_detail_retains_it(tmp_path):
    store = Store(tmp_path)
    folder = tmp_path / "pending"
    folder.mkdir()
    (folder / "data.json").write_text(json.dumps({"data": "result"}))
    metadata = {
        "history": [{"metrics/mAP50(B)": "0"}],
        "versions": {"code": "sha"},
        "dataset_id": "dataset-a",
        "project": {"dataset": {"name": "set-a"}},
    }
    a = store.publish(folder, "train", "key", metadata)
    with TestClient(create_app(tmp_path)) as client:
        brief = client.get("/api/artifacts").json()[0]
        assert "files" not in brief["manifest"]
        assert "history" not in brief["manifest"]["metadata"]
        assert brief["manifest"]["metadata"]["epoch_count"] == 1
        full = client.get(f"/api/artifacts/{a['id']}").json()
        assert full["manifest"]["metadata"]["history"] == metadata["history"]
        assert "data.json" in full["manifest"]["files"]


def test_lost_training_worker_cleans_working_images_but_keeps_checkpoint(tmp_path):
    store = Store(tmp_path)
    folder = tmp_path / "dataset-input"
    folder.mkdir()
    (folder / "data.yaml").write_text("train: images/train\n")
    data = store.publish(folder, "dataset", "data", {})
    job = enqueue(store, "train", {"dataset_id": data["id"]})
    training = store.state / "jobs" / job["id"] / "training"
    images = training / "input-dataset/images"
    images.mkdir(parents=True)
    (images / "image.png").write_bytes(b"temporary image")
    checkpoint = training / "resume-last.pt"
    checkpoint.write_bytes(b"complete epoch")
    store.update_job(job["id"], status="running", pid=99999999)
    Supervisor(store).tick()
    assert store.job(job["id"])["status"] == "interrupted"
    assert not (training / "input-dataset").exists()
    assert checkpoint.read_bytes() == b"complete epoch"
