import sys
from types import SimpleNamespace

from boardgen.studio import environment
from boardgen.studio.preparation import frame_key
from boardgen.studio.runtime import stage_versions


def test_relocated_code_is_fingerprinted_and_invalidates_its_stage(monkeypatch, tmp_path):
    monkeypatch.setattr(environment, "TOOLS", tmp_path)
    monkeypatch.setattr(environment, "blender", lambda: str(tmp_path / "blender"))
    monkeypatch.setattr(environment, "spirula", lambda: tmp_path / "spirula")
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(try_to_load_from_cache=lambda *a: None)
    )
    versions = environment.fingerprint()
    for stage, path in [
        ("cameras", "camera/camera_audit.py"),
        ("geometry", "evaluation/benchmark_tools.py"),
        ("appearance", "camera/camera_audit.py"),
    ]:
        original = stage_versions(versions, stage)
        assert original["code"][path] == versions["code"][path]
        changed = {**versions, "code": {**versions["code"], path: "changed"}}
        assert stage_versions(changed, stage) != original
    for path in ("video/views.py", "common/manifest.py", "rendering/blender_scene.py"):
        assert path in versions["code"]


def test_relocated_mask_code_invalidates_frame_cache(tmp_path):
    from boardgen.studio.models import Figure, Recipe
    from boardgen.studio.storage import Store

    (tmp_path / "video.mp4").write_bytes(b"video fixture")
    store = Store(tmp_path)
    figure = Figure(id="a", name="A", video="video.mp4")
    recipe = Recipe()
    versions = {"code": {"video/segment.py": "original"}, "neural": {}, "packages": {}}
    original = frame_key(store, figure, recipe, versions)
    versions["code"]["video/segment.py"] = "changed"
    assert frame_key(store, figure, recipe, versions) != original
