import numpy as np
import pytest

from boardgen.methods.mv.mv_adapter import ordered_views, pipeline_class, view_embeddings


def test_view_embeddings_reference():
    omega = 1.0 / 10000 ** (np.arange(768, dtype=np.float64) / 768)
    expected = np.stack([np.r_[np.sin(i * omega), np.cos(i * omega)] for i in range(4)])
    np.testing.assert_allclose(
        view_embeddings(["front", "left", "back", "right"], 1536), expected, atol=3e-8
    )
    assert ordered_views({"right": "r", "front": "f", "back": "b"}) == [
        ("front", "f"),
        ("back", "b"),
        ("right", "r"),
    ]
    with pytest.raises(ValueError):
        ordered_views({"top": "t"})


@pytest.mark.gpu
def test_conditioner_concatenates_views_as_tokens(tmp_path):
    mx = pytest.importorskip("mlx.core", reason="MLX conditioning requires Apple Silicon")
    from pathlib import Path

    if not Path("tmp/benchmark_tools/hunyuan3d-swift/python/shape/hy3dmlx").is_dir():
        pytest.skip("Pinned MLX port is not installed; run make setup-tools")
    from PIL import Image

    image = tmp_path / "view.png"
    Image.new("RGBA", (32, 32), (120, 120, 120, 255)).save(image)
    spec = {"type": "DinoImageEncoderMV", "kwargs": {"config": {"hidden_size": 1536}}}
    pipeline = pipeline_class()(
        lambda _: mx.ones((1, 1370, 1536)),
        None,
        None,
        {"conditioner": {"params": {"main_image_encoder": spec}}},
        mx.float32,
    )
    result = np.array(pipeline.encode_image({"back": image, "front": image}))
    assert result.shape == (2, 2740, 1536)
    assert np.count_nonzero(result[1]) == 0
    vectors = 1 + view_embeddings(["front", "back"], 1536)
    np.testing.assert_allclose(result[0, 0], vectors[0])
    np.testing.assert_allclose(result[0, 1369], vectors[0])
    np.testing.assert_allclose(result[0, 1370], vectors[1])
    np.testing.assert_allclose(result[0, -1], vectors[1])
    spec["type"] = "DinoImageEncoder"
    with pytest.raises(ValueError, match="actual pretrained MV"):
        pipeline.encode_image({"front": image})
