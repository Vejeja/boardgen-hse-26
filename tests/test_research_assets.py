import copy
import json
from pathlib import Path

import pytest

from boardgen.maintenance import verify_research_assets as ASSETS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def preserved():
    bundle = ROOT / "research/published-experiment"
    return (
        json.loads((ROOT / "research/yolo/results.json").read_text()),
        json.loads((bundle / "snapshot.json").read_text()),
        json.loads((bundle / "manifest.json").read_text()),
    )


def test_preserved_index_matches_published_experiment(preserved):
    ASSETS.check_index(*preserved)


@pytest.mark.parametrize("missing", ["model", "label", "glb", "initial", "last", "mapping"])
def test_preserved_index_rejects_incomplete_or_wrong_binaries(preserved, missing):
    record, snapshot, manifest = copy.deepcopy(preserved)
    files = manifest["files"]
    if missing == "model":
        snapshot["model_sha256"] = "0" * 64
    elif missing == "glb":
        snapshot["selected_figures"][0]["mesh_sha256"] = "0" * 64
    elif missing == "mapping":
        manifest["weight_names"].clear()
    else:
        suffix = {"label": ".txt", "initial": "yolo11s.pt", "last": "last.pt"}[missing]
        name = next(name for name in files if name.endswith(suffix))
        del files[name]
    with pytest.raises(ValueError):
        ASSETS.check_index(record, snapshot, manifest)


@pytest.fixture
def metadata_root(tmp_path, preserved):
    _, _, manifest = preserved
    paths = [
        "research/yolo/results.json",
        "research/3d-quality/manifest.json",
        *[
            str(p.relative_to(ROOT))
            for p in (ROOT / "research/published-experiment").glob("*.json")
        ],
        *manifest["repository_files"],
    ]
    for name in set(paths):
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.hardlink_to(ROOT / name)
    return tmp_path


def test_metadata_verification_works_without_local_weights_snapshot_or_gallery(metadata_root):
    ASSETS.verify(metadata_root)
    with pytest.raises(ValueError, match="Local snapshot parts are missing"):
        ASSETS.verify(metadata_root, full=True)


def test_metadata_verification_rejects_corrupted_local_weights(metadata_root):
    weights = metadata_root / "research/yolo/weights/best.pt"
    weights.parent.mkdir()
    weights.write_bytes(b"corrupted weights")
    with pytest.raises(ValueError, match="Model SHA"):
        ASSETS.verify(metadata_root)
