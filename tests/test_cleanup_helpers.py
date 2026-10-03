import hashlib

import numpy as np
import pytest

from boardgen.common.image_utils import largest_component
from boardgen.common.manifest import sha256_file
from boardgen.common.utils import file_sha256
from boardgen.evaluation.benchmark3d import file_hash
from boardgen.studio.storage import sha
from boardgen.video.segment import _dominant
from boardgen.video.turntable import _dominant_component


def test_file_hashes_agree_for_binary_input_and_unicode_path(tmp_path):
    payload = bytes(range(256)) * 9000
    path = tmp_path / "кадры.bin"
    path.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert (
        file_sha256(path, 7)
        == sha256_file(str(path), 17)
        == file_hash(path)
        == sha(path)
        == expected
    )
    assert sha256_file(str(tmp_path / "missing")) is None
    with pytest.raises(FileNotFoundError):
        sha(tmp_path / "missing")


def test_localisation_and_segmentation_keep_only_the_largest_component():
    mask = np.zeros((20, 20), np.uint8)
    mask[1:3, 1:3] = 1
    mask[10:13, 10:14] = 1
    expected = np.zeros_like(mask)
    expected[10:13, 10:14] = 1
    for select in [largest_component, _dominant, _dominant_component]:
        np.testing.assert_array_equal(select(mask), expected)
        assert select(np.zeros_like(mask)) is None
