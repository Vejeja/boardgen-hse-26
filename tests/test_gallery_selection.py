import json
from pathlib import Path

import pytest
import trimesh
import yaml
from fastapi.testclient import TestClient
from PIL import Image

from boardgen.studio import environment
from boardgen.studio.api import create_app
from boardgen.studio.datasets import build
from boardgen.studio.selection import available_packages, import_selection
from boardgen.studio.stages import selected_assets
from boardgen.studio.storage import Store, json_write, sha


@pytest.fixture
def inputs(tmp_path):
    repo = tmp_path / "repository"
    package = repo / "out/reviewed"
    gallery = package / "gallery"
    gallery.mkdir(parents=True)
    (repo / "metadata").mkdir()
    (repo / "metadata/video_index.csv").write_text(
        "\ufeffvideo,class_id,yolo_class,ru_name\n"
        'obj_02.mp4,1,shadow_dragon_master,"Теневой дракон (красный, вожак)"\n'
        "obj_19.mp4,18,grisban_the_thirsty,Грисбан Жаждущий\n"
    )
    videos = repo / "research/3d-quality/videos"
    videos.mkdir(parents=True)
    choices, catalog = [], []
    for obj in ["obj_02", "obj_19"]:
        video = videos / f"{obj}.mp4"
        video.write_bytes(obj.encode())
        mesh = package / obj / "photo.glb"
        mesh.parent.mkdir()
        trimesh.creation.box().export(mesh)
        candidate = {
            "id": obj + "-candidate",
            "mesh": f"../{obj}/photo.glb",
            "mesh_sha256": sha(mesh),
            "label": "Выбранный вариант",
            "note": "Ограничения сохранены",
        }
        json_write(
            gallery / f"{obj}.json",
            {"object": obj, "source_video_sha256": sha(video), "candidates": [candidate]},
        )
        catalog.append({"object": obj, "file": f"{obj}.json"})
        choices.append(
            {
                "object": obj,
                "id": candidate["id"],
                "name": "Название из браузера не является справочником",
                "mesh_sha256": sha(mesh),
                "source_video_sha256": sha(video),
            }
        )
    json_write(gallery / "comparison-catalog.json", catalog)
    json_write(package / "completion-report.json", {})
    return repo, package, {"schema_version": 1, "choices": choices}


def test_subset_keeps_selected_mesh_and_remaps_global_id(inputs, tmp_path):
    repo, package, content = inputs
    content["choices"] = content["choices"][1:]
    target = tmp_path / "selected"
    result = import_selection(content, package, target, repository=repo)
    assert result["class_count"] == 1
    row = result["class_map"]["figures"][0]
    assert (row["local_class_id"], row["source_class_id"], row["name"]) == (
        0,
        18,
        "Грисбан Жаждущий",
    )
    store = Store(target)
    project = store.load()
    assert [f.id for f in project.figures] == ["obj_19"]
    assert project.dataset.real_sources == []
    assets = selected_assets(store, project, {})
    assert set(assets) == {"obj_19"}
    item = assets["obj_19"]
    assert sha(Path(item["directory"]) / "figure.glb") == content["choices"][0]["mesh_sha256"]
    assert store.jobs() == []


def test_preserved_gallery_is_discovered_without_local_output(tmp_path):
    package = tmp_path / "research/3d-quality"
    (package / "gallery").mkdir(parents=True)
    json_write(package / "gallery/comparison-catalog.json", [])
    json_write(package / "completion-report.json", {})
    assert available_packages(tmp_path) == [package.resolve()]


def test_manifest_without_local_gallery_is_not_offered_as_package(tmp_path):
    package = tmp_path / "research/3d-quality"
    package.mkdir(parents=True)
    json_write(package / "manifest.json", {"files": []})
    assert available_packages(tmp_path) == []


def test_preserved_gallery_cannot_redirect_outside_repository(tmp_path):
    repository = tmp_path / "repository"
    (repository / "research").mkdir(parents=True)
    external = tmp_path / "external"
    (external / "gallery").mkdir(parents=True)
    json_write(external / "gallery/comparison-catalog.json", [])
    json_write(external / "completion-report.json", {})
    (repository / "research/3d-quality").symlink_to(external, target_is_directory=True)
    assert available_packages(repository) == []


@pytest.mark.parametrize(
    "change", ["hash", "video", "candidate", "duplicate", "empty", "escape", "provenance"]
)
def test_invalid_selection_does_not_create_project(inputs, tmp_path, change):
    repo, package, content = inputs
    if change == "hash":
        content["choices"][0]["mesh_sha256"] = "0" * 64
    elif change == "video":
        (repo / "research/3d-quality/videos/obj_02.mp4").write_bytes(b"different recording")
    elif change == "candidate":
        content["choices"][0]["id"] = "unknown"
    elif change == "duplicate":
        content["choices"].append(content["choices"][0])
    elif change == "empty":
        content["choices"] = []
    elif change == "escape":
        path = package / "gallery/obj_02.json"
        data = json.loads(path.read_text())
        data["candidates"][0]["mesh"] = "../../../escape.glb"
        json_write(path, data)
    else:
        path = package / "gallery/obj_02.json"
        data = json.loads(path.read_text())
        data["candidates"][0]["coordinate_provenance"] = "source.json"
        json_write(path, data)
        json_write(package / "gallery/source.json", {"changed": True})
        json_write(package / "gallery/comparison-assets.json", {"source.json": "0" * 64})
    target = tmp_path / "rejected"
    with pytest.raises(ValueError):
        import_selection(content, package, target, repository=repo)
    assert not target.exists()


def test_import_never_overwrites_existing_project(inputs, tmp_path):
    repo, package, content = inputs
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep.txt").write_text("original")
    with pytest.raises(ValueError, match="новый каталог"):
        import_selection(content, package, target, repository=repo)
    assert list(target.iterdir()) == [target / "keep.txt"]


def test_dataset_publishes_subset_ids_and_exact_asset_hash(inputs, tmp_path):
    repo, package, content = inputs
    content["choices"] = content["choices"][1:]
    target = tmp_path / "subset"
    import_selection(content, package, target, repository=repo)
    store = Store(target)
    records = []
    for i, color in enumerate(["red", "blue"]):
        image = tmp_path / f"{i}.png"
        Image.new("RGB", (80, 80), color).save(image)
        records.append(
            {
                "image": str(image),
                "labels": [[0, 0.5, 0.5, 0.25, 0.25]],
                "group": str(i),
                "origin": "synthetic",
            }
        )
    destination = tmp_path / "dataset"
    destination.mkdir()
    build(store, store.load(), destination, records)
    data = yaml.safe_load((destination / "data.yaml").read_text())
    assert data["names"] == {0: "Грисбан Жаждущий"}
    mapping = json.loads((destination / "class-map.json").read_text())["figures"]
    assert len(mapping) == 1
    assert (mapping[0]["local_class_id"], mapping[0]["source_class_id"]) == (0, 18)
    assert mapping[0]["mesh_sha256"] == content["choices"][0]["mesh_sha256"]
    assert {int(p.read_text().split()[0]) for p in destination.glob("labels/*/*.txt")} == {0}


def test_api_import_switches_to_fresh_project_without_jobs(inputs, tmp_path, monkeypatch):
    repo, package, content = inputs
    monkeypatch.setattr(environment, "REPOSITORY", repo)
    original = tmp_path / "original"
    app = create_app(original)
    before = sha(original / "project.yaml")
    with TestClient(app) as client:
        state = client.get("/api/project").json()
        assert (
            client.post(
                "/api/figure-selection", json={"selection": content, "package": str(package)}
            ).status_code
            == 403
        )
        assert client.get("/api/figure-selection/packages").json() == [str(package)]
        response = client.post(
            "/api/figure-selection",
            headers={"X-Studio-Token": state["token"], "X-Workspace": state["root"]},
            json={"selection": content, "package": str(package)},
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["root"] != state["root"]
        assert data["selection_import"]["class_count"] == 2
        assert client.get("/api/jobs").json()["total"] == 0
    assert sha(original / "project.yaml") == before


def test_api_import_does_not_interrupt_active_project(inputs, tmp_path, monkeypatch):
    repo, package, content = inputs
    monkeypatch.setattr(environment, "REPOSITORY", repo)
    app = create_app(tmp_path / "active")
    with TestClient(app) as client:
        state = client.get("/api/project").json()
        monkeypatch.setattr(app.state.context.store, "jobs", lambda: [{"status": "running"}])
        response = client.post(
            "/api/figure-selection",
            headers={"X-Studio-Token": state["token"]},
            json={"selection": content, "package": str(package)},
        )
        assert response.status_code == 409
        assert client.get("/api/project").json()["root"] == state["root"]
        assert not (repo / "workspaces").exists()


def test_candidate_library_reuses_verified_assets_without_changing_choice(inputs, tmp_path):
    from boardgen.studio.selection import import_candidates

    repo, package, content = inputs
    destination = tmp_path / "library"
    import_selection(content, package, destination, repository=repo)
    store = Store(destination)
    before = store.load().model_dump()
    count = len(store.artifacts("figure"))

    gallery = package / "gallery/obj_02.json"
    data = json.loads(gallery.read_text())
    mesh = package / "obj_02/second.glb"
    trimesh.creation.icosphere().export(mesh)
    data["candidates"].append(
        {
            "id": "second",
            "mesh": "../obj_02/second.glb",
            "mesh_sha256": sha(mesh),
            "label": "Другой вариант",
            "note": "Проверить",
        }
    )
    json_write(gallery, data)
    first = import_candidates(store, package, repo)
    assert len(store.artifacts("figure")) == count + 1
    assert store.load().model_dump() == before
    second = import_candidates(Store(destination), package, repo)
    assert second == first
    assert Store(destination).load().model_dump() == before
    mesh.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="сумма"):
        import_candidates(store, package, repo)
    assert len(store.artifacts("figure")) == count + 1
