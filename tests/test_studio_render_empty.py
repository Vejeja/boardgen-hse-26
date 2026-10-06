import pytest
from PIL import Image

from boardgen.studio.models import Figure, Project
from boardgen.studio.rendering import render
from boardgen.studio.storage import Store


@pytest.mark.parametrize("negative", [True, False])
def test_empty_annotations_keep_negatives_and_resample_positives(tmp_path, monkeypatch, negative):
    store = Store(tmp_path)
    board = tmp_path / "board.png"
    Image.new("RGB", (16, 16)).save(board)
    project = Project(
        figures=[Figure(id="a", name="A", video="a.mp4")],
        scenes={
            "boards": [{"image": str(board), "width_m": 0.5, "height_m": 0.5}],
            "negative_fraction": 0.5 if negative else 0.0,
            "seed": 0,
            "objects_per_scene": [1, 1],
            "max_attempts": 2,
        },
    )
    assets = {
        "a": {
            "id": "asset-a",
            "directory": str(tmp_path),
            "manifest": {
                "metadata": {"appearance": "photo", "mesh": "a.glb", "footprint_over_height": 1}
            },
        }
    }
    (tmp_path / "a.glb").write_bytes(b"mock asset")
    calls = []

    def fake_render(scene, scene_path, label_path, **kwargs):
        calls.append(scene)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("" if len(calls) == 1 else "0 0.5 0.5 0.5 0.5\n")

    monkeypatch.setattr("boardgen.rendering.blender_render.render_scene", fake_render)
    records = render(store, project, tmp_path / "renders", assets, 1)
    assert len(calls) == (1 if negative else 2)
    assert records[0]["labels"] == ([] if negative else [[0, 0.5, 0.5, 0.5, 0.5]])
