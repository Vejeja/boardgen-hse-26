import numpy as np
from PIL import Image


def conditioning_image(source, alpha_threshold=128, foreground_gain=1.0):
    if not 1 <= alpha_threshold <= 254 or not 0.4 <= foreground_gain <= 1.2:
        raise ValueError("Invalid conditioning alpha threshold or foreground gain")
    pixels = np.array(Image.open(source).convert("RGBA"))
    foreground = pixels[:, :, 3] >= alpha_threshold
    if not foreground.any():
        raise ValueError("Empty conditioning foreground")
    pixels[:, :, 3] = np.where(foreground, 255, 0)

    pixels[:, :, :3] = np.clip(pixels[:, :, :3].astype(float) * foreground_gain, 0, 255).astype(
        "uint8"
    )
    return Image.fromarray(pixels)
