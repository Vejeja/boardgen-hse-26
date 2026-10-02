import cv2
import numpy as np

from boardgen.video.neuralseg import recover_warm_foreground


def test_warm_recovery_restores_base_and_preserves_background_and_holes():
    lab = np.full((80, 100, 3), [180, 128, 133], np.uint8)
    lab[15:55, 40:60, 2] = 148
    lab[55:70, 20:80, 2] = 145
    lab[30:36, 46:54, 2] = 133
    alpha = np.zeros((80, 100), np.uint8)
    alpha[15:55, 40:60] = 255
    alpha[30:36, 46:54] = 0
    alpha[20, 40] = 220
    recovered, metadata = recover_warm_foreground(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), alpha)
    assert metadata["applied"]
    assert (recovered[57:68, 22:78] == 255).all()
    assert not recovered[:10].any()
    assert not recovered[31:35, 47:53].any()
    assert (recovered >= alpha).all()


def test_warm_recovery_skips_foreground_without_colour_separation():
    image = np.full((80, 100, 3), 180, np.uint8)
    alpha = np.zeros((80, 100), np.uint8)
    alpha[15:55, 40:60] = 220
    recovered, metadata = recover_warm_foreground(image, alpha)
    assert not metadata["applied"]
    assert np.array_equal(recovered, alpha)
