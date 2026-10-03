import numpy as np
import pytest
from PIL import Image

from boardgen.methods.mv.conditioning import conditioning_image


def test_faint_alpha_tails_do_not_expand_conditioning_foreground(tmp_path):
    pixels = np.full((20, 30, 4), 220, dtype="uint8")
    pixels[:, :, 3] = 1
    pixels[5:15, 10:20, 3] = 240
    source = tmp_path / "photo.png"
    Image.fromarray(pixels).save(source)
    before = source.read_bytes()
    result = np.array(conditioning_image(source, foreground_gain=0.65))
    ys, xs = np.where(result[:, :, 3])
    assert (xs.min(), ys.min(), xs.max(), ys.max()) == (10, 5, 19, 14)
    assert np.all(result[5:15, 10:20, :3] == 143)
    assert source.read_bytes() == before


def test_empty_mask_cannot_enter_generation(tmp_path):
    source = tmp_path / "empty.png"
    Image.new("RGBA", (10, 10), (200, 200, 200, 1)).save(source)
    with pytest.raises(ValueError, match="Empty conditioning foreground"):
        conditioning_image(source)
