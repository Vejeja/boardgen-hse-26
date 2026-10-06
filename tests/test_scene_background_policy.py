import pytest
from pydantic import ValidationError

from boardgen.studio.datasets import audit_backgrounds
from boardgen.studio.models import Project
from boardgen.studio.rendering import figure_heights, negative_slots, render
from boardgen.studio.storage import Store, json_write
from boardgen.studio.training import guard_evaluation_split


def scene(split, board="board-a", sky="sky-a"):
    return {
        "origin": "synthetic",
        "fixed_split": split,
        "board_sha256": board,
        "skybox_sha256": sky,
    }


def test_test_backgrounds_must_be_independent_even_when_images_differ():
    with pytest.raises(ValueError, match="Источник фона"):
        audit_backgrounds([scene("train"), scene("test", sky="new-sky")], "test_only")


def test_train_val_may_share_backgrounds_when_test_is_held_out():
    report = audit_backgrounds(
        [scene("train"), scene("val"), scene("test", "board-b", "sky-b")], "test_only"
    )
    assert report["unique_sources"] == {"board_sha256": 2, "skybox_sha256": 2}
    with pytest.raises(ValueError, match="Источник фона"):
        audit_backgrounds([scene("train"), scene("val")], "all_splits")


def test_strict_background_policy_requires_source_provenance():
    with pytest.raises(ValueError, match="происхождения"):
        audit_backgrounds([{"origin": "synthetic", "fixed_split": "test"}], "test_only")


def test_old_projects_keep_board_generation_and_no_test_split():
    project = Project()
    assert project.scenes.scene_type == "board"
    assert project.dataset.test_fraction == 0


def test_mask_scaling_uses_only_chosen_figures_and_an_available_reference(tmp_path, monkeypatch):
    import cv2
    import numpy as np

    store = Store(tmp_path)
    project = Project.model_validate(
        {
            "figures": [{"id": k, "name": k, "video": "unused.mp4"} for k in ["a", "b", "c"]],
            "scenes": {
                "selected_classes": ["b", "c"],
                "scale_basis": "shared_video",
                "reference_class": "a",
                "height_overrides": {"a": 0.2, "b": 0.04},
            },
        }
    )
    assets = {}
    for figure_id, height in [("b", 3), ("c", 6)]:
        folder = tmp_path / ("frames-" + figure_id)
        masks = folder / "views/reference/masks"
        masks.mkdir(parents=True)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:height, :] = 255
        cv2.imwrite(str(masks / "view_0000.png"), mask)
        json_write(
            folder / "views/dataset.json",
            {
                "fit": [0],
                "crop_xyxy": [0, 0, 10, 10],
                "source_video": "unused.mp4",
            },
        )
        frame = store.publish(folder, "frames", figure_id, {}, figure_id)
        assets[figure_id] = {"manifest": {"metadata": {"frames_id": frame["id"]}}}

    class Capture:
        def get(self, key):
            assert key == cv2.CAP_PROP_FRAME_HEIGHT
            return 10

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: Capture())
    heights, provenance = figure_heights(store, project, assets)
    assert heights == pytest.approx({"b": 0.04, "c": 0.08})
    assert provenance["reference"] == "b"
    assert set(provenance["mask_height_fraction_of_frame"]) == {"b", "c"}


def test_stratified_negatives_cover_both_scene_types_and_all_batches():
    slots = negative_slots(80, 0.1, 42, mixed=True)
    assert len(slots) == 8
    assert sum(i % 2 == 0 for i in slots) == 4
    batches = [negative_slots(10, 0.1, 42 + b * 100003, mixed=True) for b in range(8)]
    assert all(len(s) == 1 for s in batches)
    assert sum(next(iter(s)) % 2 == 0 for s in batches) == 4


def test_invalid_platform_camera_and_split_fractions_are_rejected():
    with pytest.raises(ValidationError):
        Project.model_validate({"scenes": {"platform": {"frame_fraction": [0.7, 0.3]}}})
    with pytest.raises(ValidationError):
        Project.model_validate({"dataset": {"val_fraction": 0.6, "test_fraction": 0.4}})


def test_new_dataset_version_cannot_bypass_test_background_guard():
    old = {"records": [{**scene("train"), "split": "train", "sha256": "a", "group": "old"}]}
    new = {
        "background_audit": {"policy": "test_only"},
        "records": [{**scene("test"), "split": "test", "sha256": "b", "group": "new"}],
    }
    with pytest.raises(ValueError, match="Источник фона"):
        guard_evaluation_split(old, new, "test")


def test_platform_scene_places_one_figure_on_top_and_excludes_podium_from_ids(
    tmp_path, monkeypatch
):
    from PIL import Image

    store = Store(tmp_path)
    board = tmp_path / "board.png"
    Image.new("RGB", (16, 16)).save(board)
    mesh = tmp_path / "a.glb"
    mesh.write_bytes(b"mock asset, Blender invocation is replaced")
    project = Project.model_validate(
        {
            "figures": [{"id": "a", "name": "A", "video": "a.mp4"}],
            "scenes": {
                "boards": [{"image": str(board)}],
                "scene_type": "platform",
                "negative_fraction": 0,
                "objects_per_scene": [2, 6],
                "balanced_classes": True,
            },
        }
    )
    asset = {
        "id": "a",
        "directory": str(tmp_path),
        "manifest": {
            "metadata": {
                "mesh": "a.glb",
                "appearance": "photo",
                "footprint_over_height": 1,
            }
        },
    }
    captured = []

    def fake_render(scene, scene_path, label_path, **kwargs):
        captured.append(scene)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("0 0.5 0.5 0.2 0.4\n")

    monkeypatch.setattr("boardgen.rendering.blender_render.render_scene", fake_render)
    rows = render(store, project, tmp_path / "render", {"a": asset}, 1)
    spec = captured[0]
    assert len(rows[0]["labels"]) == len(spec["objects"]) == 1
    assert spec["objects"][0]["xy"] == [0, 0]
    assert spec["flags"]["mesh_hover_m"] == pytest.approx(spec["platform"]["height_m"] + 0.00005)
    assert "pass_index" not in spec["platform"]
    assert spec["camera"]["target_m"][2] > spec["platform"]["height_m"]


def test_background_identity_uses_content_and_rejects_copies_across_splits(tmp_path):
    from PIL import Image

    from boardgen.studio.backgrounds import identity, validate_sources
    from boardgen.studio.storage import sha

    a, b, sky = tmp_path / "a.png", tmp_path / "b.png", tmp_path / "sky.png"
    for path, color in [(a, "red"), (b, "blue"), (sky, "green")]:
        Image.new("RGB", (16, 16), color).save(path)
    assert identity(a) == {"sha256": sha(a), "source_id": "image:" + sha(a)}
    assert identity(a)["source_id"] != identity(b)["source_id"]
    assert identity(a)["sha256"] != identity(b)["sha256"]
    project = Project(
        scenes={
            "boards": [{"image": str(a)}],
            "skyboxes": [str(a)],
            "split_sources": {"test": {"boards": [{"image": str(b)}], "skyboxes": [str(sky)]}},
        },
        dataset={"test_fraction": 0.2, "background_policy": "test_only"},
    )
    validate_sources(Store(tmp_path), project)
    b.write_bytes(a.read_bytes())
    assert identity(a) == identity(b)
    with pytest.raises(ValueError, match="пересекает"):
        validate_sources(Store(tmp_path), project)


def test_no_test_split_still_uses_separate_validation_sources(tmp_path, monkeypatch):
    from PIL import Image

    store = Store(tmp_path)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    for p, color in [(a, "red"), (b, "blue")]:
        Image.new("RGB", (16, 16), color).save(p)
    mesh = tmp_path / "a.glb"
    mesh.write_bytes(b"fixture")
    project = Project(
        figures=[{"id": "a", "name": "A", "video": "a.mp4"}],
        scenes={
            "boards": [{"image": str(a)}],
            "objects_per_scene": [1, 1],
            "negative_fraction": 0,
            "split_sources": {"val": {"boards": [{"image": str(b)}]}},
        },
        dataset={"val_fraction": 0.5, "test_fraction": 0},
    )
    asset = {
        "id": "a",
        "directory": str(tmp_path),
        "manifest": {
            "metadata": {"mesh": "a.glb", "appearance": "photo", "footprint_over_height": 1}
        },
    }

    def fake(scene, scene_path, label_path, **kwargs):
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("0 .5 .5 .2 .4\n")

    monkeypatch.setattr("boardgen.rendering.blender_render.render_scene", fake)
    rows = render(store, project, tmp_path / "render", {"a": asset}, 2)
    assert [(r["fixed_split"], r["scene"]["board"]["image"]) for r in rows] == [
        ("train", str(a)),
        ("val", str(b)),
    ]


def test_exact_overhead_camera_has_finite_position():
    import math
    import random

    from boardgen.studio.rendering import camera

    project = Project(scenes={"camera": {"elevation_deg": [90, 90], "jitter_xy_m": 0}})
    result = camera(project.scenes.camera, random.Random(2))
    assert all(math.isfinite(v) for v in result["position_m"])
    assert result["position_m"][:2] == pytest.approx([0, 0], abs=1e-12)
