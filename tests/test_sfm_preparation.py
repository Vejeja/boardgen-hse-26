import json
from pathlib import Path

import pytest

from boardgen.evaluation import benchmark_tools


@pytest.fixture
def sfm_dataset(tmp_path, monkeypatch):
    (tmp_path / "dataset.json").write_text(json.dumps({"fit": [0, 1], "heldout": [2]}))
    commands = []

    def run(out, command, expected, **kwargs):
        commands.append(command)
        for name in expected:
            path = Path(name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"colmap-fixture")
        return {"status": "success", "runtime_seconds": 0}

    monkeypatch.setattr(benchmark_tools, "run_command", run)
    return tmp_path, commands


def test_partial_native_registration_is_rejected(sfm_dataset, monkeypatch):
    root, _ = sfm_dataset
    monkeypatch.setattr(benchmark_tools, "read_colmap", lambda path: [("view_0000.png",)])
    with pytest.raises(RuntimeError, match="exactly the fit set"):
        benchmark_tools.sfm(root, Path("spirula"), features="sift")
    assert not (root / "train/sparse/0").exists()
    record = json.loads((root / "runs/sfm-aliked/run.json").read_text())
    assert record["status"] == "failed"
    assert record["registered_views"] == 1


def test_sensitive_sift_keeps_pnp_acceptance_and_excludes_heldout(sfm_dataset, monkeypatch):
    root, commands = sfm_dataset
    monkeypatch.setattr(
        benchmark_tools, "read_colmap", lambda path: [("view_0000.png",), ("view_0001.png",)]
    )
    record = benchmark_tools.sfm(
        root,
        Path("spirula"),
        name="sensitive",
        features="sift",
        max_features=16384,
        peak_threshold=0.001,
        edge_threshold=20,
        match_ratio=0.85,
    )
    flags = commands[0]
    assert flags[flags.index("--peak-threshold") + 1] == "0.001"
    assert flags[flags.index("--max-features") + 1] == "16384"
    assert "--min-pnp-inliers" not in flags
    assert "--min-pnp-ratio" not in flags
    assert record["registered_views"] == 2
    assert (root / "train/sparse/0/cameras.bin").is_file()
    assert "heldout" not in " ".join(flags)
