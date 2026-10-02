import math

import numpy as np
import pytest

from boardgen.camera.camera_model import (
    CameraIntrinsics,
    CropResizeTransform,
    TurntableCameraModel,
    TurntableView,
)


@pytest.mark.unit
def test_crop_resize_transform_updates_intrinsics():
    intr = CameraIntrinsics(width=4000, height=3000, fx=2000.0, fy=2100.0, cx=1999.5, cy=1499.5)
    transform = CropResizeTransform(
        source_width=4000,
        source_height=3000,
        crop_xyxy=(1000.0, 500.0, 3000.0, 2500.0),
        output_width=1000,
        output_height=1000,
    )

    out = transform.transform_intrinsics(intr)

    assert out.width == 1000
    assert out.height == 1000
    assert out.fx == pytest.approx(1000.0)
    assert out.fy == pytest.approx(1050.0)
    assert out.cx == pytest.approx(499.75)
    assert out.cy == pytest.approx(499.75)


@pytest.mark.unit
def test_projection_maps_turntable_origin_to_principal_point():
    intr = CameraIntrinsics(width=640, height=480, fx=500.0, fy=500.0, cx=319.5, cy=239.5)
    model = TurntableCameraModel(
        image_width=640,
        image_height=480,
        intrinsics=intr,
        elevation_deg=20.0,
        camera_distance=2.0,
        views=[TurntableView(frame_index=0, theta_rad=0.0)],
    )

    p = model.projection_matrices()[0]
    hp = p @ np.array([0.0, 0.0, 0.0, 1.0])
    uv = hp[:2] / hp[2]

    assert uv[0] == pytest.approx(intr.cx)
    assert uv[1] == pytest.approx(intr.cy)


@pytest.mark.unit
def test_turntable_rotation_changes_off_axis_projection():
    intr = CameraIntrinsics(width=640, height=480, fx=500.0, fy=500.0, cx=319.5, cy=239.5)
    model = TurntableCameraModel(
        image_width=640,
        image_height=480,
        intrinsics=intr,
        elevation_deg=10.0,
        camera_distance=2.0,
        views=[
            TurntableView(frame_index=0, theta_rad=0.0),
            TurntableView(frame_index=10, theta_rad=math.pi / 2.0),
        ],
    )

    point = np.array([0.2, 0.0, 0.2, 1.0])
    uv = []
    for p in model.projection_matrices():
        hp = p @ point
        uv.append(hp[:2] / hp[2])

    assert abs(float(uv[0][0] - uv[1][0])) > 10.0


@pytest.mark.unit
def test_camera_model_yaml_round_trip(tmp_path):
    model = TurntableCameraModel(
        image_width=16,
        image_height=12,
        intrinsics=CameraIntrinsics.from_fov(16, 12, 45.0),
        views=[TurntableView(frame_index=3, theta_rad=0.25, view_id="view_0000")],
        confidence=0.75,
    )
    path = tmp_path / "camera.yaml"

    model.save_yaml(str(path))
    loaded = TurntableCameraModel.load_yaml(str(path))

    assert loaded.schema_version == model.schema_version
    assert loaded.views[0].frame_index == 3
    assert loaded.projection_matrices()[0].shape == (3, 4)


@pytest.mark.unit
def test_project_points_matches_analytic_zero_elevation_camera():
    intr = CameraIntrinsics(width=101, height=81, fx=100.0, fy=120.0, cx=50.0, cy=40.0)
    model = TurntableCameraModel(
        image_width=101,
        image_height=81,
        intrinsics=intr,
        camera_distance=2.0,
        views=[TurntableView(frame_index=0, theta_rad=0.0)],
    )
    uv, depth = model.project_points(np.array([[0.2, 0.0, 0.1]]))

    assert depth[0] == pytest.approx(2.0)
    assert uv[0, 0] == pytest.approx(60.0)
    assert uv[0, 1] == pytest.approx(34.0)


@pytest.mark.unit
def test_crop_projection_obeys_pixel_center_affine_transform():
    transform = CropResizeTransform(400, 300, (100.0, 50.0, 300.0, 250.0), 100, 100)
    model = TurntableCameraModel(
        image_width=100,
        image_height=100,
        intrinsics=CameraIntrinsics(400, 300, 200.0, 200.0, 199.5, 149.5),
        crop_resize_transform=transform,
        views=[TurntableView(0, 0.0)],
        camera_distance=2.0,
    )
    uv, _ = model.project_points(np.array([[0.2, 0.0, 0.1]]))
    full = np.array([219.5, 139.5])

    assert uv[0] == pytest.approx((full - np.array([100.0, 50.0])) * 0.5)


@pytest.mark.unit
def test_validation_rejects_duplicate_frames_and_unknown_schema():
    model = TurntableCameraModel(
        10,
        10,
        CameraIntrinsics.from_fov(10, 10, 45),
        [TurntableView(1, 0.0), TurntableView(1, 1.0)],
    )
    with pytest.raises(ValueError, match="duplicate"):
        model.validate()
    with pytest.raises(ValueError, match="unsupported camera model schema"):
        TurntableCameraModel.from_dict({"schema_version": "future"})


@pytest.mark.unit
def test_v1_schema_migrates_explicitly():
    model = TurntableCameraModel(
        10,
        10,
        CameraIntrinsics.from_fov(10, 10, 45),
        [TurntableView(0, 0.0)],
    )
    data = model.to_dict()
    data["schema_version"] = "boardgen.turntable_camera_model.v1"
    data["method"] = data.pop("initialization_method")
    data["intrinsics"]["distortion_model"] = "none"

    migrated = TurntableCameraModel.from_dict(data)

    assert migrated.schema_version == "boardgen.turntable_camera_model.v2"
    assert migrated.intrinsics.distortion_model == "none_assumed"
