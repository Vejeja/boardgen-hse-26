import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import trimesh
from pydantic import ValidationError

from boardgen.localization.assets import load_figure
from boardgen.localization.fitting import box_from_points, fit_figure
from boardgen.localization.geometry import Camera, blender_camera, board_corners, estimate_camera
from boardgen.localization.pipeline import Frame, Localizer, scaled_intrinsics
from boardgen.localization.registration import register_board
from boardgen.maintenance.localization_benchmark import select_frames
from boardgen.studio.models import Localization, LocalizationCamera, Project
from boardgen.studio.storage import Store


def camera():
    return blender_camera(
        {"position_m": [0.15, -0.7, 0.8], "roll_deg": 5, "fov_deg": 45}, [1000, 800]
    )


def test_camera_matches_independent_blender_export():
    fixture = json.loads((Path(__file__).parent / "fixtures/localization_blender.json").read_text())
    for row in fixture:
        result = blender_camera(row["spec"], row["image_size"])
        np.testing.assert_allclose(result.K, row["K"], atol=0.0002)
        np.testing.assert_allclose(result.R, row["R"], atol=3e-7)
        np.testing.assert_allclose(result.t, row["t"], atol=3e-7)


@pytest.mark.parametrize("distortion", [(), (0.05, -0.01, 0.001, -0.002, 0)])
def test_pose_from_corners_and_ray_intersection_are_metric(distortion):
    original = camera()
    original = Camera(
        original.width, original.height, original.K, original.R, original.t, distortion
    )
    world = board_corners(0.5, 0.4)
    found, quality = estimate_camera(
        original.K, distortion, (1000, 800), original.project(world), 0.5, 0.4
    )
    assert quality["reprojection_rmse_px"] < 1e-6
    np.testing.assert_allclose(found.R, original.R, atol=1e-6)
    np.testing.assert_allclose(found.t, original.t, atol=1e-6)
    point = [0.07, -0.08, 0]
    np.testing.assert_allclose(
        found.plane_point(original.project([point])[0]), point[:2], atol=1e-7
    )


def test_invalid_correspondences_and_calibration_fail_explicitly():
    c = camera()
    corners = c.project(board_corners(0.5, 0.4))
    with pytest.raises(ValueError, match="выпуклый"):
        estimate_camera(c.K, (), (1000, 800), corners[[0, 2, 1, 3]], 0.5, 0.4)
    with pytest.raises(ValidationError):
        LocalizationCamera(fx=float("nan"))
    with pytest.raises(ValidationError):
        LocalizationCamera(distortion=[0, 0, 0])
    with pytest.raises(ValidationError):
        Localization(corners=[[0, 0], [1, 0]])
    with pytest.raises(ValueError, match="Пропорции"):
        scaled_intrinsics(LocalizationCamera(), (1000, 800))
    k = scaled_intrinsics(LocalizationCamera(), (576, 576))
    assert k[0, 0] == pytest.approx(1390.587 / 2)
    assert k[0, 2] == 288


def test_glb_graph_transform_and_asymmetric_base_are_preserved(tmp_path):
    base = trimesh.creation.cylinder(radius=0.2, height=0.04)
    base.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    base.apply_translation([-0.3, 0.02, 0])
    body = trimesh.creation.box(extents=[1.0, 1.0, 0.4])
    body.apply_translation([0, 0.54, 0])
    mesh = trimesh.util.concatenate([base, body])
    scene = trimesh.Scene()
    scene.add_geometry(mesh, transform=trimesh.transformations.translation_matrix([3, 7, -2]))
    path = tmp_path / "figure.glb"
    scene.export(path)
    original = path.read_bytes()
    g = load_figure(path, 0.104)
    assert g.height_m == 0.104
    assert g.base_xy[0] == pytest.approx(-0.03, abs=0.002)
    assert g.radius_m == pytest.approx(0.02, abs=0.001)
    np.testing.assert_allclose(g.base_at([0.1, 0.2], 90), [0.1 - g.base_xy[1], 0.2 + g.base_xy[0]])
    assert path.read_bytes() == original


def test_fit_recovers_position_for_exact_bbox_without_claiming_yaw(tmp_path):
    mesh = trimesh.creation.box(extents=[0.3, 1.0, 0.5])
    path = tmp_path / "figure.glb"
    mesh.export(path)
    g = load_figure(path, 0.05)
    c = camera()
    xy, yaw = [0.07, -0.04], 63
    bbox = box_from_points(c.project(g.placed(xy, yaw)))
    found = fit_figure(c, g, bbox, [0.5, 0.4])
    np.testing.assert_allclose(found["xy_m"], xy, atol=0.001)
    assert found["bbox_fit_error"] < 0.001
    assert found["yaw_observed"] is False
    assert found["radius_m"] == g.radius_m


def test_template_registration_uses_image_not_oracle_pose():
    rng = np.random.default_rng(23)
    template = np.zeros((500, 625, 3), np.uint8) + 180
    for _ in range(220):
        x, y = rng.integers([0, 0], [625, 500])
        cv2.circle(
            template,
            (int(x), int(y)),
            int(rng.integers(2, 9)),
            tuple(int(v) for v in rng.integers(0, 255, 3)),
            -1,
        )
    c = camera()
    world = board_corners(0.5, 0.4)
    source = np.float32([[0, 0], [625, 0], [625, 500], [0, 500]])
    h = cv2.getPerspectiveTransform(source, c.project(world).astype(np.float32))
    image = cv2.warpPerspective(template, h, (1000, 800))
    found, quality = register_board(image, template, c.K, (), [0.5, 0.4])
    assert quality["inliers"] >= 10
    assert np.max(np.linalg.norm(found.project(world) - c.project(world), axis=1)) < 2


def test_registration_failure_never_returns_oracle_or_stale_coordinates():
    c = camera()
    settings = Localization(
        camera=LocalizationCamera(
            width=1000, height=800, fx=c.K[0, 0], fy=c.K[1, 1], cx=500, cy=400
        )
    )
    settings.board.width_m = 0.5
    settings.board.height_m = 0.4
    localizer = Localizer(settings, np.zeros((200, 200, 3), np.uint8), {})
    result, found = localizer.process(
        Frame(np.zeros((800, 1000, 3), np.uint8), Path("image.png"), 0, None), []
    )
    assert found is None
    assert result["status"] == "registration_failed"
    assert "camera" not in result and result["figures"] == []


def test_subset_requires_all_same_class_matches_and_no_extra_detections():
    record = {"image": "test/a.png", "scene_type": "board", "labels": [[0, 0.5, 0.5, 0.1, 0.1]]}
    prediction = {
        "image": "a.png",
        "classes": [0],
        "boxes": [[0.5, 0.5, 0.1, 0.1]],
        "confidence": [0.9],
    }
    assert len(select_frames([record], [prediction])) == 1
    prediction["classes"] = [1]
    assert select_frames([record], [prediction]) == []
    prediction.update(classes=[0, 0], boxes=[[0.5, 0.5, 0.1, 0.1]] * 2, confidence=[0.9, 0.8])
    assert select_frames([record], [prediction]) == []


def test_localization_map_is_collected_and_rebased_in_snapshots():
    from boardgen.studio.transfer import rebase_project, source_paths

    project = Project().model_dump(mode="json")
    assert "" not in source_paths(project)
    project["localization"]["board"]["image"] = "map.png"
    assert "map.png" in source_paths(project)
    rebased = rebase_project(
        project,
        {"source_project": "/old/project", "weight_names": {}},
        [("/old/project", "/new/project")],
    )
    assert rebased["localization"]["board"]["image"] == "/new/project/map.png"


def test_preview_requires_token_and_rejects_external_file(tmp_path):
    from fastapi.testclient import TestClient

    from boardgen.studio.api import create_app

    image = tmp_path / "photo.png"
    cv2.imwrite(str(image), np.zeros((20, 30, 3), np.uint8))
    external = tmp_path.parent / "external-preview.png"
    cv2.imwrite(str(external), np.zeros((20, 30, 3), np.uint8))
    with TestClient(create_app(tmp_path)) as client:
        data = client.get("/api/project").json()
        headers = {"X-Studio-Token": data["token"], "X-Workspace": data["root"]}
        assert (
            client.post("/api/localization-preview", json={"path": str(image)}).status_code == 403
        )
        response = client.post(
            "/api/localization-preview", json={"path": str(image)}, headers=headers
        )
        assert response.status_code == 200
        assert response.json()["width"] == 30
        assert response.json()["image"].startswith("data:image/jpeg;base64,")
        assert (
            client.post(
                "/api/localization-preview", json={"path": str(external)}, headers=headers
            ).status_code
            == 422
        )


def test_localization_stage_publishes_frozen_model_and_coordinates(tmp_path, monkeypatch):
    from boardgen.studio import localization, training

    store = Store(tmp_path / "project")
    mesh_folder = tmp_path / "mesh"
    mesh_folder.mkdir()
    trimesh.creation.box(extents=[0.3, 1, 0.5]).export(mesh_folder / "figure.glb")
    asset = store.publish(mesh_folder, "figure", "mesh", {"mesh": "figure.glb"}, "a")
    project = Project(figures=[{"id": "a", "name": "A", "video": "unused.mp4"}])
    project.scenes.asset_versions = {"a": asset["id"]}
    dataset_folder = tmp_path / "dataset"
    dataset_folder.mkdir()
    (dataset_folder / "dataset.json").write_text(
        json.dumps({"records": [{"scene": {"objects": [{"class_id": 0, "height_m": 0.05}]}}]})
    )
    dataset = store.publish(
        dataset_folder,
        "dataset",
        "data",
        {"names": ["A"], "project": project.model_dump(mode="json")},
    )
    weights_folder = tmp_path / "weights"
    weights_folder.mkdir()
    (weights_folder / "best.pt").write_bytes(b"test checkpoint")
    model = store.publish(
        weights_folder, "train", "weights", {"dataset_id": dataset["id"], "best": "best.pt"}
    )
    c = camera()
    settings = project.localization
    settings.camera = LocalizationCamera(
        width=1000, height=800, fx=c.K[0, 0], fy=c.K[1, 1], cx=500, cy=400
    )
    settings.board.width_m = 0.5
    settings.board.height_m = 0.4
    settings.registration = "corners"
    settings.corners = (c.project(board_corners(0.5, 0.4)) / [1000, 800]).tolist()
    source = store.root / "photo.png"
    cv2.imwrite(str(source), np.zeros((800, 1000, 3), np.uint8))
    settings.board.image = str(source)
    g = load_figure(Path(asset["directory"]) / "figure.glb", 0.05)
    bbox = box_from_points(c.project(g.placed([0.05, 0.01], 30))) / [1000, 800, 1000, 800]

    class Tensor:
        def __init__(self, value):
            self.value = value

        def cpu(self):
            return self

        def tolist(self):
            return self.value

    prediction = SimpleNamespace(
        boxes=SimpleNamespace(cls=Tensor([0]), conf=Tensor([0.95]), xywhn=Tensor([bbox.tolist()]))
    )
    fake_model = SimpleNamespace(names={0: "A"}, predict=lambda *a, **k: [prediction])
    monkeypatch.setattr(
        training,
        "trained_model",
        lambda *a: (fake_model, model, Path(model["directory"]) / "best.pt"),
    )
    monkeypatch.setattr(training, "device", lambda: "cpu")
    project.scenes.asset_versions = {}
    folder = tmp_path / "job"
    folder.mkdir()
    result = localization.localize(
        store, project, {"model_id": model["id"], "source": str(source)}, folder, {}
    )
    published = store.artifact(result["artifacts"][0])
    root = Path(published["directory"])
    scene = json.loads((root / "scene.json").read_text())
    assert scene["models"]["0"]["asset_id"] == asset["id"]
    assert scene["frames"][0]["status"] == "processed"
    assert len((root / "frames.jsonl").read_text().splitlines()) == 1
    assert (root / "models/0.glb").read_bytes() == (
        Path(asset["directory"]) / "figure.glb"
    ).read_bytes()
    np.testing.assert_allclose(scene["frames"][0]["figures"][0]["xy_m"], [0.05, 0.01], atol=0.002)
