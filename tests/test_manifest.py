import json

import pytest

from boardgen.common.manifest import build_manifest, write_manifest


@pytest.mark.unit
def test_manifest_contains_hashes_and_versions(tmp_path):
    cfg = tmp_path / "config.yaml"
    video = tmp_path / "clip.mp4"
    cfg.write_text("camera: {}\n")
    video.write_bytes(b"not really a video")

    manifest = build_manifest(
        config_paths=[str(cfg)],
        source_videos=[str(video)],
        components={"segmentation": {"backend": "classic"}},
        random_seed=123,
    )
    out = tmp_path / "run_manifest.json"
    write_manifest(str(out), manifest)
    loaded = json.loads(out.read_text())

    assert loaded["schema_version"] == "boardgen.run_manifest.v1"
    assert loaded["config_hashes"][str(cfg)]
    assert loaded["source_video_hashes"][str(video)]
    assert loaded["components"]["segmentation"]["backend"] == "classic"
    assert loaded["random_seed"] == 123
