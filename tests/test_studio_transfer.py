import json
import os
import shutil
import subprocess

import pytest
import yaml

from boardgen.studio import environment
from boardgen.studio.jobs import enqueue
from boardgen.studio.models import Project
from boardgen.studio.storage import Store, sha
from boardgen.studio.transfer import export_project, restore_project, verify_bundle


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    repo = tmp_path / "repository"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "video.mp4").write_bytes(b"fixture input video")
    subprocess.run(["git", "-C", str(repo), "add", "video.mp4"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    tools = tmp_path / "tools"
    (tools / "yolo").mkdir(parents=True)
    (tools / "yolo/yolo11n.pt").write_bytes(b"fixture initial weights")
    monkeypatch.setattr(environment, "TOOLS", tools)
    store = Store(tmp_path / "original")
    store.grant(str(repo))
    store.save(
        Project(figures=[{"id": "figure-a", "name": "Figure", "video": str(repo / "video.mp4")}])
    )
    pending = tmp_path / "pending"
    pending.mkdir()
    content = os.urandom(4096)
    (pending / "best.pt").write_bytes(content)
    (pending / "last.pt").write_bytes(content)
    (pending / "data.yaml").write_text(
        yaml.safe_dump({"path": str(store.state / "artifacts"), "train": "images/train"})
    )
    artifact = store.publish(
        pending,
        "train",
        "fixture-key",
        {"best": "best.pt", "last": "last.pt", "source": str(repo / "video.mp4")},
    )
    dataset_folder = tmp_path / "dataset-input"
    dataset_folder.mkdir()
    (dataset_folder / "data.yaml").write_text("train: images/train\n")
    data = store.publish(dataset_folder, "dataset", "dataset-key", {})
    job = enqueue(store, "train", {"dataset_id": data["id"]})
    folder = store.state / "jobs" / job["id"] / "training"
    folder.mkdir()
    (folder / "resume-last.pt").write_bytes(content)
    (folder / "input-dataset").mkdir()
    (folder / "input-dataset/disposable.png").write_bytes(b"working copy")
    (folder.parent / "process.log").write_text("historic log\n")
    store.update_job(job["id"], status="cancelled", pid=99999999, cancel=1)
    bundle = tmp_path / "bundle"
    index = export_project(store, bundle, repo, part_bytes=1024)
    return repo, store, bundle, index, artifact, job, content


def test_transfer_restores_references_and_resume_history_without_mutating_binaries(snapshot):
    repo, original, bundle, index, artifact, job, content = snapshot

    clone = repo / "clone"
    clone.mkdir()
    shutil.copyfile(repo / "video.mp4", clone / "video.mp4")
    destination = clone / "workspace"
    old_manifest = sha(original.state / artifact["path"] / "manifest.json")
    report = restore_project(bundle, destination, clone)
    restored = Store(destination)
    assert report["artifacts"] == 2 and report["jobs"] == 1
    assert restored.load().figures[0].video == str(clone / "video.mp4")
    assert (
        restored.resolve(restored.load().training.weights, exists=True).read_bytes()
        == b"fixture initial weights"
    )
    result = restored.artifact(artifact["id"])
    assert result["manifest"]["relocation"]["source_manifest_sha256"] == old_manifest
    assert (restored.state / artifact["path"] / "best.pt").read_bytes() == content
    config = yaml.safe_load((restored.state / artifact["path"] / "data.yaml").read_text())
    assert config["path"] == str(destination / ".boardgen/artifacts")
    stopped = restored.job(job["id"])
    assert stopped["pid"] is None and stopped["status"] == "cancelled"
    assert stopped["snapshot"]["figures"][0]["video"] == str(clone / "video.mp4")
    assert (restored.state / "jobs" / job["id"] / "training/resume-last.pt").read_bytes() == content
    assert not (restored.state / "jobs" / job["id"] / "training/input-dataset").exists()
    assert original.artifact(artifact["id"])["manifest"]["metadata"]["source"] == str(
        repo / "video.mp4"
    )
    assert len({v["sha256"] for v in index["files"].values()}) < len(index["files"])
    assert len(index["parts"]) > 1


def test_corrupt_part_is_rejected_before_restoring_project(snapshot, tmp_path):
    repo, _, bundle, index, *_ = snapshot
    part = bundle / index["parts"][0]["name"]
    data = bytearray(part.read_bytes())
    data[-1] ^= 1
    part.write_bytes(data)
    destination = tmp_path / "new"
    with pytest.raises(ValueError, match="Повреждена часть"):
        restore_project(bundle, destination, repo)
    assert not destination.exists()


def test_changed_repository_inputs_and_existing_destination_are_rejected(snapshot, tmp_path):
    repo, _, bundle, _, *_ = snapshot
    destination = tmp_path / "new"
    (repo / "video.mp4").write_bytes(b"changed")
    with pytest.raises(ValueError, match="изменён вход"):
        restore_project(bundle, destination, repo)
    assert not destination.exists()
    destination.mkdir()
    (destination / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="новую"):
        restore_project(bundle, destination, repo)
    assert (destination / "keep.txt").read_text() == "keep"


def test_active_jobs_cannot_be_exported_or_automatically_restored(snapshot, tmp_path):
    repo, store, bundle, index, _, job, _ = snapshot
    store.update_job(job["id"], status="queued")
    with pytest.raises(ValueError, match="остановите задания"):
        export_project(store, tmp_path / "active", repo)
    index["jobs"][0]["status"] = "queued"
    (bundle / "manifest.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="активные задания"):
        restore_project(bundle, tmp_path / "new", repo)


def test_archive_paths_cannot_escape_project(snapshot):
    _, _, bundle, index, *_ = snapshot
    index["files"]["../escape.txt"] = index["files"]["project.yaml"]
    (bundle / "manifest.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="Недопустимый путь"):
        verify_bundle(bundle)


def test_fresh_project_does_not_require_downloading_weights_for_transfer(snapshot, tmp_path):
    repo, *_ = snapshot
    (environment.TOOLS / "yolo/yolo11n.pt").unlink()
    fresh = Store(tmp_path / "fresh")
    bundle = tmp_path / "fresh-bundle"
    export_project(fresh, bundle, repo)
    restored_path = tmp_path / "fresh-restored"
    restore_project(bundle, restored_path, repo)
    assert Store(restored_path).load().training.weights == "yolo11n.pt"
