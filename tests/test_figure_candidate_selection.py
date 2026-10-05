import json

import pytest
import trimesh

from boardgen.common.figure_names import figure_names
from boardgen.evaluation.benchmark3d import file_hash
from boardgen.methods.repair.select_figure_candidate import select


def candidates(tmp_path):
    asset, candidate = tmp_path / "obj_15/asset", tmp_path / "candidate"
    asset.mkdir(parents=True)
    (candidate / "validation/score").mkdir(parents=True)
    for root in [asset, candidate]:
        trimesh.creation.box().export(root / "photo.glb")
        for name in ["preview-photo.jpg", "neutral.jpg"]:
            (root / name).write_bytes(b"saved diagnostic preview")
    old = {"iou": 0.92, "foreground_recall": 0.95}
    meta = {
        "object": "obj_15",
        "selected": "refined",
        "source_video_sha256": "original-video",
        "files": {"photo.glb": file_hash(asset / "photo.glb")},
        "validation": {"refined": {"validation": old}},
        "settings": {},
        "refinement": {"scores": {}},
    }
    (asset / "asset.json").write_text(json.dumps(meta))
    score = {"candidate_sha256": file_hash(candidate / "photo.glb"), "summary": {"validation": old}}
    (candidate / "validation/score/metrics.json").write_text(json.dumps(score))
    (candidate / "result.json").write_text(
        json.dumps(
            {
                "photo_sha256": file_hash(candidate / "photo.glb"),
                "generation_provenance": {
                    "source_video_sha256": "original-video",
                    "method": "four-view",
                    "parameters": {},
                    "weights": {},
                    "inputs": [],
                    "port_commit": "pinned",
                },
                "assessment": "Generated shape prior",
            }
        )
    )
    return asset, candidate


def test_wrong_source_video_is_rejected_before_archiving(tmp_path):
    asset, candidate = candidates(tmp_path)
    p = candidate / "result.json"
    data = json.loads(p.read_text())
    data["generation_provenance"]["source_video_sha256"] = "different-video"
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="different source video"):
        select(asset, tmp_path, candidate, "mv", "reviewed")
    assert (asset / "photo.glb").exists()
    assert not (asset.parent / "archive").exists()


def test_large_shape_regression_preserves_current_model(tmp_path):
    asset, candidate = candidates(tmp_path)
    p = candidate / "validation/score/metrics.json"
    data = json.loads(p.read_text())
    data["summary"]["validation"]["foreground_recall"] = 0.8
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="regresses silhouette"):
        select(asset, tmp_path, candidate, "mv", "reviewed")
    assert (asset / "photo.glb").exists()
    assert not (asset.parent / "archive").exists()


def test_selected_generated_geometry_keeps_its_own_settings(tmp_path):
    asset, candidate = candidates(tmp_path)
    result = select(asset, tmp_path, candidate, "mv", "four views checked")
    assert result["selected_method"] == "hunyuan_mv"
    assert result["settings"]["selected_candidate"]["method"] == "four-view"
    assert "Generated shape prior" in result["refinement"]["assessment"]


def test_names_cover_exact_stable_class_ids_without_duplicate_downloads():
    names = figure_names()
    assert set(names) == {f"obj_{i:02}" for i in range(1, 27)}
    assert len({value["slug"] for value in names.values()}) == 26
    assert all(not value["name"].startswith("obj_") for value in names.values())


def test_approximation_override_cannot_accept_a_collapsed_shape(tmp_path):
    asset, candidate = candidates(tmp_path)
    p = candidate / "validation/score/metrics.json"
    data = json.loads(p.read_text())
    data["summary"]["validation"]["iou"] = 0.48
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="IoU >=0.7"):
        select(asset, tmp_path, candidate, "mv", "reviewed", accept_generated_approximation=True)
    assert not (asset.parent / "archive").exists()


def test_explicit_approximation_records_the_shape_regression(tmp_path):
    asset, candidate = candidates(tmp_path)
    p = candidate / "validation/score/metrics.json"
    data = json.loads(p.read_text())
    data["summary"]["validation"].update(iou=0.74, foreground_recall=0.9)
    p.write_text(json.dumps(data))
    result = select(
        asset,
        tmp_path,
        candidate,
        "mv",
        "complete approximate body reviewed",
        accept_generated_approximation=True,
    )
    assert result["refinement"]["explicit_generated_approximation_selection"] is True
    assert result["refinement"]["silhouette_change"]["iou"] == pytest.approx(-0.18)
    assert result["validation"]["refined"]["validation"]["iou"] == 0.74


def paired_candidate(tmp_path):
    asset, candidate = candidates(tmp_path)
    provenance = {
        "photo_sha256": file_hash(candidate / "photo.glb"),
        "template_provenance": {
            "target_video_sha256": "original-video",
            "source_object": "obj_16",
            "method": "paired_observed_surface",
        },
        "assessment": "Observed red counterpart, shared sculpt assumption",
    }
    (candidate / "result.json").write_text(json.dumps(provenance))
    return asset, candidate


def test_wrong_counterpart_is_rejected_before_archiving(tmp_path):
    asset, candidate = paired_candidate(tmp_path)
    path = candidate / "result.json"
    data = json.loads(path.read_text())
    data["template_provenance"]["source_object"] = "obj_18"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="different figure"):
        select(asset, tmp_path, candidate, "paired", "reviewed", accept_paired_surface=True)
    assert (asset / "photo.glb").exists()
    assert not (asset.parent / "archive").exists()


def test_paired_selection_keeps_observed_geometry_provenance(tmp_path):
    asset, candidate = paired_candidate(tmp_path)
    path = candidate / "validation/score/metrics.json"
    score = json.loads(path.read_text())
    score["summary"]["validation"].update(iou=0.81, foreground_recall=0.85)
    path.write_text(json.dumps(score))
    selected = select(
        asset,
        tmp_path,
        candidate,
        "paired",
        "Open leg spaces checked against photos",
        accept_paired_surface=True,
    )
    assert selected["selected_method"] == "paired_observed_surface"
    assert selected["settings"]["selected_candidate"]["source_object"] == "obj_16"
    assert selected["refinement"]["explicit_paired_surface_selection"] is True
    assert selected["refinement"]["explicit_generated_approximation_selection"] is False


def test_stale_atlas_coordinate_receipt_is_rejected_before_archiving(tmp_path):
    asset, candidate = candidates(tmp_path)
    p = candidate / "result.json"
    provenance = json.loads(p.read_text())
    provenance["photo_sha256"] = "unblended-atlas-hash"
    p.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="coordinate provenance"):
        select(asset, tmp_path, candidate, "mv", "reviewed")
    assert (asset / "photo.glb").exists()
    assert not (asset.parent / "archive").exists()
