import json

import pytest

from boardgen.common.manifest import sha256_file
from boardgen.video.turntable import ClipInfo, probe


def test_probe_cache_distinguishes_equal_basenames(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()
    videos = [tmp_path / "short/video.mp4", tmp_path / "full/video.mp4"]
    for i, video in enumerate(videos):
        video.parent.mkdir()
        video.write_bytes(bytes([i]))
        info = ClipInfo(
            str(video), 100 + i, 60, 3840, 2160, 60, 0.5, 1, (0, 0, 20, 20), (0, 0, 20, 20)
        )
        key = cache / f"video_{sha256_file(str(video))}_probe.json"
        key.write_text(json.dumps(info.to_json()))
    monkeypatch.setattr(
        "boardgen.video.turntable.read_probe_stack", lambda *args: pytest.fail("cache miss")
    )
    assert [probe(str(p), str(cache)).n_frames for p in videos] == [100, 101]
