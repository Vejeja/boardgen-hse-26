import cv2
import numpy as np


def largest_component(mask: np.ndarray) -> np.ndarray | None:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return None
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == label).astype(np.uint8)


def load_view_masks(views_dir: str) -> tuple[np.ndarray, list[str]]:
    import os

    paths = sorted(
        os.path.join(views_dir, name)
        for name in os.listdir(views_dir)
        if name.startswith("view_") and name.endswith(".png")
    )
    masks = []
    for path in paths:
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"cannot read mask {path}; camera/view alignment must be preserved")
        if image.ndim == 3:
            channel = (
                image[:, :, 3] if image.shape[2] == 4 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            )
        else:
            channel = image
        masks.append((channel > 127).astype(np.uint8))
    if not masks:
        raise ValueError(f"no masks in {views_dir}")
    return np.stack(masks), paths
