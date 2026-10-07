import json
from pathlib import Path

import pytest
import trimesh
from fastapi.testclient import TestClient

from boardgen.studio import candidates, comparison, environment
from boardgen.studio.api import create_app
from boardgen.studio.jobs import enqueue
from boardgen.studio.models import Figure, Project
from boardgen.studio.parameters import validate
from boardgen.studio.stages import selected_assets
from boardgen.studio.storage import Store, json_write, sha


@pytest.fixture
def project_store(tmp_path):
    store = Store(tmp_path)
    (tmp_path / "video.mp4").write_bytes(b"source")
    project = Project(
        figures=[
            Figure(
                id="obj_01",
                name="Фигура",
                video="video.mp4",
                reviewed_views={"front": 0, "left": 1, "back": 2, "right": 3},
                front_view=0,
                period_views=20,
                focal_px=800,
            )
        ]
    )
    store.save(project)
    return store, project


def publish_figure(store, key):
    folder = store.root / key
    folder.mkdir()
    trimesh.creation.box().export(folder / "figure.glb")
    return store.publish(
        folder,
        "figure",
        key,
        {"mesh": "figure.glb", "source_sha256": sha(store.root / "video.mp4")},
        "obj_01",
    )


def test_variants_are_unique_and_native_precedes_fusion():
    values = validate(
        "prepare-candidates", {"variants": ["spirula_depth_fusion", "spirula_native"]}
    )
    assert values["variants"] == ["spirula_native", "spirula_depth_fusion"]
    with pytest.raises(ValueError):
        validate("prepare-candidates", {"variants": ["mv", "mv"]})


def test_frame_cache_tracks_view_preparation_but_not_training_code(project_store):
    from boardgen.studio.preparation import frame_key

    store, project = project_store
    figure = project.figures[0]
    recipe = project.prepare.resolved(figure.id)
    versions = {
        "code": {"video/views.py": "original", "studio/training.py": "original"},
        "neural": {},
        "packages": {},
    }
    original = frame_key(store, figure, recipe, versions)
    versions["code"]["studio/training.py"] = "changed"
    assert frame_key(store, figure, recipe, versions) == original
    versions["code"]["video/views.py"] = "changed"
    assert frame_key(store, figure, recipe, versions) != original


def test_series_checks_all_required_views_before_enqueue(project_store):
    store, project = project_store
    project.figures[0].reviewed_views = {}
    store.save(project)
    with pytest.raises(ValueError, match="четыре ракурса"):
        enqueue(store, "prepare-candidates", {})
    assert not store.jobs()


def test_one_failure_keeps_other_candidates_and_snapshot(project_store, monkeypatch):
    store, project = project_store
    job = enqueue(store, "prepare-candidates", {})
    folder = store.state / "jobs" / job["id"]
    before = project.model_dump()
    calls = []

    def prepare(store, configured, figure, folder, versions):
        recipe = configured.prepare.resolved(figure.id)
        calls.append((recipe.shape.method, recipe.shape.extraction))
        if recipe.shape.method == "mv":
            raise RuntimeError("MV failed")
        return publish_figure(store, recipe.shape.extraction)

    monkeypatch.setattr(candidates, "prepare", prepare)
    with pytest.raises(RuntimeError, match="успешные версии сохранены"):
        candidates.prepare_candidates(store, project, job, folder, {})
    result = store.job(job["id"])["result"]
    assert calls == [("mv", "native"), ("spirula", "native"), ("spirula", "depth_fusion")]
    assert len(result["artifacts"]) == 2
    assert [r["status"] for r in result["candidates"]] == ["failed", "success", "success"]
    assert project.model_dump() == before == store.load().model_dump()
    assert json.loads((folder / "candidate-series.json").read_text()) == result


def test_cancel_does_not_discard_published_candidate(project_store, monkeypatch):
    store, project = project_store
    job = enqueue(store, "prepare-candidates", {})
    calls = []

    def prepare(store, configured, figure, folder, versions):
        calls.append(configured.prepare.resolved(figure.id).shape.method)
        item = publish_figure(store, "completed")
        store.update_job(job["id"], cancel=1)
        return item

    monkeypatch.setattr(candidates, "prepare", prepare)
    with pytest.raises(KeyboardInterrupt):
        candidates.prepare_candidates(store, project, job, store.state / "jobs" / job["id"], {})
    assert calls == ["mv"]
    assert len(store.job(job["id"])["result"]["artifacts"]) == 1
    assert len(Store(store.root).artifacts("figure")) == 1


def test_choice_survives_restart_and_new_preparation(project_store):
    store, project = project_store
    first = publish_figure(store, "chosen")
    app = create_app(store.root)
    client = TestClient(app)
    data = client.get("/api/project").json()
    headers = {"X-Studio-Token": data["token"], "X-Workspace": data["root"]}
    response = client.put(
        "/api/candidate-choice",
        headers=headers,
        json={"figure_id": "obj_01", "artifact_id": first["id"], "revision": data["revision"]},
    )
    assert response.status_code == 200, response.text
    publish_figure(store, "newer")
    restored = Store(store.root)
    assert selected_assets(restored, restored.load(), {})["obj_01"]["id"] == first["id"]
    assert restored.load().scenes.selected_classes == ["obj_01"]
    stale = client.put(
        "/api/candidate-choice",
        headers=headers,
        json={"figure_id": "obj_01", "artifact_id": None, "revision": data["revision"]},
    )
    assert stale.status_code == 422
    current = response.json()
    removed = client.put(
        "/api/candidate-choice",
        headers=headers,
        json={"figure_id": "obj_01", "artifact_id": None, "revision": current["revision"]},
    )
    assert removed.status_code == 200
    assert Store(store.root).load().scenes.selected_classes == []
    assert len(restored.artifacts("figure")) == 2


def test_corrupt_chosen_mesh_does_not_fall_back(project_store):
    store, project = project_store
    item = publish_figure(store, "chosen")
    project.scenes.asset_versions["obj_01"] = item["id"]
    (Path(item["directory"]) / "figure.glb").write_bytes(b"changed")
    publish_figure(store, "other")
    with pytest.raises(ValueError, match="повреждён"):
        selected_assets(store, project, {})


def test_gallery_choice_is_durable_and_manifest_limits_file_access(project_store, monkeypatch):
    store, project = project_store
    repo = store.root / "repository"
    package = repo / "out/package"
    gallery = package / "gallery"
    gallery.mkdir(parents=True)
    model = package / "model.glb"
    model.write_bytes(b"model")
    choice = {
        "object": "obj_01",
        "id": "model",
        "mesh_sha256": sha(model),
        "source_video_sha256": sha(store.root / "video.mp4"),
    }
    json_write(gallery / "comparison-assets.json", {"../model.glb": sha(model)})
    json_write(gallery / "comparison-catalog.json", [{"object": "obj_01", "file": "figure.json"}])
    json_write(
        gallery / "figure.json",
        {
            "object": "obj_01",
            "source_video_sha256": choice["source_video_sha256"],
            "frames": [],
            "candidates": [{**choice, "mesh": "../model.glb"}],
        },
    )
    json_write(package / "completion-report.json", {})
    monkeypatch.setattr(environment, "REPOSITORY", repo)
    identifier = next(iter(comparison.packages()))
    url = comparison.file_url(identifier, "../model.glb")
    assert "..%2Fmodel.glb" in url

    json_write(store.state / "gallery-choices.json", {identifier: [choice]})
    assert comparison.gallery_selection(Store(store.root), identifier)["choices"] == [choice]
    with pytest.raises(ValueError, match="реестре"):
        comparison.package_file(identifier, "../../video.mp4")
    assert not hasattr(comparison, "save_gallery_selection")


def test_series_reuses_new_inputs_even_when_previous_cache_is_disabled(project_store, monkeypatch):
    from boardgen.studio.preparation import input_cached, remember_input, series_inputs

    store, project = project_store
    project.prepare.reuse_completed = False
    store.save(project)
    job = enqueue(store, "prepare-candidates", {})
    seen = []
    previous = publish_figure(store, "shared-input")

    def prepare(store, configured, figure, folder, versions):
        shared = input_cached(store, "shared-input", configured.prepare.reuse_completed)
        if shared is None:
            shared = publish_figure(store, "shared-input")
            remember_input(shared)
        seen.append(shared["id"])
        return publish_figure(
            store, configured.prepare.resolved(figure.id).shape.extraction + str(len(seen))
        )

    monkeypatch.setattr(candidates, "prepare", prepare)
    candidates.prepare_candidates(store, project, job, store.state / "jobs" / job["id"], {})
    assert len(seen) == 3 and len(set(seen)) == 1
    assert seen[0] != previous["id"]
    assert series_inputs.get() is None


@pytest.mark.parametrize("reuse_completed", [True, False])
def test_native_and_depth_fusion_share_representation_without_mutating_it(
    project_store, monkeypatch, reuse_completed
):
    import shutil

    import numpy as np

    from boardgen.methods.orbit import constrain_surface, run_candidates
    from boardgen.studio import preparation

    store, project = project_store
    figure = project.figures[0]
    versions = {"code": {}, "tools": {}, "packages": {}}
    frame = {
        "id": "frames",
        "cache_key": "frames",
        "manifest": {"metadata": {"source_sha256": sha(store.root / "video.mp4")}},
    }
    temporary = store.root / "camera"
    (temporary / "dataset").mkdir(parents=True)
    json_write(temporary / "dataset/dataset.json", {})
    camera = store.publish(temporary, "cameras", "camera", {"root": "dataset"}, figure.id)
    monkeypatch.setattr(preparation, "camera_stage", lambda *a: camera)
    runs = []

    def run(obj, profile, iterations, cap, *, dataset, **kwargs):
        runs.append(dataset)
        output = dataset / "runs" / f"spirula-{profile}-{iterations}" / "review"
        output.mkdir(parents=True)
        trimesh.creation.box().export(output / "photo.glb")
        json_write(output / "asset.json", {"transform": np.eye(4).tolist()})

    def command(module, arguments):
        output = Path(arguments[-1])
        output.mkdir()
        if module.endswith("regularize_depth"):
            trimesh.creation.box().export(output / "mesh.glb")

    monkeypatch.setattr(run_candidates, "run", run)
    monkeypatch.setattr(preparation, "command", command)
    monkeypatch.setattr(
        constrain_surface, "constrain", lambda root, source, target: shutil.copy2(source, target)
    )
    folder = store.root / "series"
    folder.mkdir()
    project.prepare.reuse_completed = reuse_completed
    token = preparation.series_inputs.set({})
    try:
        native = preparation.geometry_stage(
            store,
            candidates.candidate_project(project, figure.id, "spirula_native"),
            figure,
            frame,
            folder,
            versions,
        )
        old_hashes = dict(native["manifest"]["files"])
        fusion = preparation.geometry_stage(
            store,
            candidates.candidate_project(project, figure.id, "spirula_depth_fusion"),
            figure,
            frame,
            folder,
            versions,
        )
        assert len(runs) == 1
        assert fusion["manifest"]["metadata"]["representation_id"] == native["id"]
        assert store.artifact(native["id"])["manifest"]["files"] == old_hashes
    finally:
        preparation.series_inputs.reset(token)
    if not reuse_completed:
        folder2 = store.root / "new-series"
        folder2.mkdir()
        token = preparation.series_inputs.set({})
        try:
            new_native = preparation.geometry_stage(
                store,
                candidates.candidate_project(project, figure.id, "spirula_native"),
                figure,
                frame,
                folder2,
                versions,
            )
            new_fusion = preparation.geometry_stage(
                store,
                candidates.candidate_project(project, figure.id, "spirula_depth_fusion"),
                figure,
                frame,
                folder2,
                versions,
            )
        finally:
            preparation.series_inputs.reset(token)
        assert len(runs) == 2
        assert new_native["id"] != native["id"]
        assert new_fusion["id"] != fusion["id"]
        assert new_fusion["manifest"]["metadata"]["representation_id"] == new_native["id"]
        store.artifact(native["id"])
        store.artifact(fusion["id"])

        folder3 = store.root / "native-only"
        folder3.mkdir()
        third_native = preparation.geometry_stage(
            store,
            candidates.candidate_project(project, figure.id, "spirula_native"),
            figure,
            frame,
            folder3,
            versions,
        )
        project.prepare.reuse_completed = True
        recovered_fusion = preparation.geometry_stage(
            store,
            candidates.candidate_project(project, figure.id, "spirula_depth_fusion"),
            figure,
            frame,
            folder3,
            versions,
        )
        assert len(runs) == 3
        assert recovered_fusion["id"] not in {fusion["id"], new_fusion["id"]}
        assert recovered_fusion["manifest"]["metadata"]["representation_id"] == third_native["id"]


def test_project_comparison_requires_identical_cameras_and_exposes_reference_files(
    project_store, monkeypatch
):
    import numpy as np
    from PIL import Image
    from test_camera_audit import sparse_fixture

    store, project = project_store
    monkeypatch.setattr(comparison, "packages", lambda: {})
    cameras = []
    for name, kind in [("one", 0), ("different", 1)]:
        folder = store.root / name
        sparse_fixture(folder / "dataset/train", kind=kind)
        json_write(folder / "dataset/dataset.json", {"source_frame_indices": [0, 10, 20, 30]})
        for directory in ("images", "masks"):
            target = folder / "dataset/reference" / directory
            target.mkdir(parents=True)
            for i in range(4):
                Image.new("RGB", (80, 60), "white").save(target / f"view_{i:04d}.png")
        cameras.append(store.publish(folder, "cameras", name, {"root": "dataset"}, "obj_01"))
    for name, camera in [
        ("native", cameras[0]),
        ("fusion", cameras[0]),
        ("incompatible", cameras[1]),
    ]:
        folder = store.root / name
        folder.mkdir()
        trimesh.creation.box().export(folder / "figure.glb")
        json_write(folder / "asset.json", {"transform": np.eye(4).tolist()})
        item = store.publish(
            folder,
            "figure",
            name,
            {
                "mesh": "figure.glb",
                "camera_id": camera["id"],
                "source_sha256": sha(store.root / "video.mp4"),
            },
            "obj_01",
        )
        if name == "native":
            project.scenes.asset_versions["obj_01"] = item["id"]
    store.save(project)
    result = comparison.project_figure(store, "obj_01")
    assert len(result["candidates"]) == 2 and len(result["unavailable"]) == 1
    assert len(result["frames"]) == 4
    with TestClient(create_app(store.root)) as client:
        assert client.get(result["frames"][0]["photo"]).status_code == 200
        response = client.get(result["frames"][0]["contour"])
        assert response.status_code == 200 and response.headers["content-type"] == "image/png"


def test_explicit_choice_required_and_preflight_does_not_create_job(project_store):
    from boardgen.studio.jobs import preflight

    store, project = project_store
    asset = publish_figure(store, "ready")
    with pytest.raises(ValueError, match="закрепите"):
        selected_assets(store, project, {})
    with pytest.raises(ValueError, match="Выберите модели"):
        preflight(store, "pilot", {})
    project.scenes.asset_versions["obj_01"] = asset["id"]
    project.scenes.scene_type = "platform"
    store.save(project)
    snapshot, parameters = preflight(store, "pilot", {})
    assert snapshot.scenes.selected_classes == ["obj_01"]
    assert store.load().scenes.selected_classes is None
    assert store.jobs() == []
    project.scenes.scale_basis = "shared_video"
    store.save(project)
    with pytest.raises(ValueError, match="размеры по маскам"):
        preflight(store, "pilot", {})
    assert store.jobs() == []
    project.scenes.selected_classes = []
    store.save(project)
    with pytest.raises(ValueError, match="Выберите модели"):
        enqueue(store, "dataset", {})
    assert store.jobs() == []


def test_first_choice_includes_only_the_chosen_class(project_store):
    store, project = project_store
    project.figures.append(Figure(id="obj_02", name="Другая", video="video.mp4"))
    store.save(project)
    item = publish_figure(store, "first")
    with TestClient(create_app(store.root)) as client:
        data = client.get("/api/project").json()
        response = client.put(
            "/api/candidate-choice",
            headers={"x-studio-token": data["token"]},
            json={"figure_id": "obj_01", "artifact_id": item["id"], "revision": data["revision"]},
        )
        assert response.status_code == 200
        assert response.json()["project"]["scenes"]["selected_classes"] == ["obj_01"]
        ready = client.post(
            "/api/preflight",
            headers={"x-studio-token": data["token"]},
            json={"kind": "dataset", "parameters": {}},
        )
        assert ready.json()["ready"] is False
        assert client.get("/api/jobs").json()["total"] == 0


def test_readiness_explains_missing_inputs_without_creating_jobs(project_store):
    store, _ = project_store
    with TestClient(create_app(store.root)) as client:
        data = client.get("/api/project").json()
        for kind, params, reason in [
            ("train", {"dataset_id": None}, "Выберите сохранённую версию датасета"),
            (
                "evaluate",
                {"dataset_id": None, "model_id": None},
                "Выберите модель и датасет с разметкой",
            ),
            (
                "predict",
                {"model_id": None, "source": ""},
                "Выберите модель и источник изображений или видео",
            ),
        ]:
            response = client.post(
                "/api/preflight",
                headers={"x-studio-token": data["token"]},
                json={"kind": kind, "parameters": params},
            )
            assert response.status_code == 200
            assert response.json() == {"ready": False, "reason": reason}
        assert client.get("/api/jobs").json()["total"] == 0
