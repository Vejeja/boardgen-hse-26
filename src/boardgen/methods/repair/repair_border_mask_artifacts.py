from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from boardgen.evaluation.benchmark3d import file_hash, write_json


def remove_verified_border_noise(alpha, max_pixels=300):
    count, labels, stats, _ = cv2.connectedComponentsWithStats((alpha >= 128).astype(np.uint8))
    if count < 2:
        raise ValueError("No foreground")
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    x, y, w, h, _ = stats[largest]
    height, width = alpha.shape
    if x == 0 or y == 0 or x + w == width or y + h == height:
        raise ValueError("Main foreground is clipped; expand the source crop")
    output = alpha.copy()
    removed = []
    for index in range(1, count):
        if index == largest:
            continue
        ix, iy, iw, ih, area = stats[index]
        border = ix == 0 or iy == 0 or ix + iw == width or iy + ih == height
        separate = ix + iw < x - 20 or ix > x + w + 20 or iy + ih < y - 20 or iy > y + h + 20
        if border and separate and area <= max_pixels:
            _, soft = cv2.connectedComponents((alpha > 0).astype(np.uint8))
            soft_ids = np.unique(soft[labels == index])
            for sid in soft_ids:
                component = soft == sid
                if (
                    sid
                    and component.sum() <= max_pixels * 4
                    and not (component & (labels == largest)).any()
                ):
                    output[component] = 0
            output[labels == index] = 0
            removed.append(
                {"bounds": [int(ix), int(iy), int(iw), int(ih)], "hard_pixels": int(area)}
            )
    return output, removed


def repair(source, destination, views):
    if destination.exists():
        raise ValueError("Use a fresh destination to preserve previous masks")
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("runs", "sparse"))
    data = json.loads((destination / "dataset.json").read_text())
    edits = []
    for index in views:
        name = f"view_{index:04d}.png"
        path = destination / "reference/masks" / name
        before = file_hash(path)
        alpha, removed = remove_verified_border_noise(cv2.imread(str(path), 0))
        if not removed:
            raise ValueError(f"No safe verified border artifact in {name}")
        cv2.imwrite(str(path), alpha)
        data["input_hashes"][str(path.relative_to(destination))] = file_hash(path)
        if index in data["fit"]:
            rgb = cv2.imread(str(destination / "reference/images" / name))
            cv2.imwrite(str(destination / "train/images" / name), np.dstack([rgb, alpha]))
            cv2.imwrite(str(destination / "train/masks" / name), alpha)
            cv2.imwrite(
                str(destination / "train/feature_masks" / name),
                (alpha >= 128).astype(np.uint8) * 255,
            )
        row = next(d for d in data["diagnostics"] if d["view"] == index)
        hard = alpha >= 128
        row.update(
            area=float(hard.mean()),
            empty=not bool(hard.any()),
            touches_border=bool(
                hard[0].any() or hard[-1].any() or hard[:, 0].any() or hard[:, -1].any()
            ),
        )
        edits.append(
            {
                "view": index,
                "original_mask_sha256": before,
                "updated_mask_sha256": file_hash(path),
                "removed": removed,
            }
        )
    data["mask_corrections"] = {
        "source_dataset": str(source.resolve()),
        "method": "visually verified isolated background-border noise only; main component and interior thin parts preserved",
        "edits": edits,
    }
    tiles = []
    for i in range(len(data["source_frame_indices"])):
        name = f"view_{i:04d}.png"
        rgb = cv2.imread(str(destination / "reference/images" / name))
        alpha = cv2.imread(str(destination / "reference/masks" / name), 0) / 255
        checker = np.indices(alpha.shape).sum(axis=0) // 16 % 2
        tile = (
            rgb * alpha[:, :, None]
            + np.where(checker[:, :, None], 180, 230) * (1 - alpha[:, :, None])
        ).astype(np.uint8)
        tile = cv2.resize(tile, (128, 128))
        cv2.putText(tile, str(i), (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        tiles.append(tile)
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(
        str(destination / "contact.png"),
        np.vstack([np.hstack(tiles[i : i + 8]) for i in range(0, len(tiles), 8)]),
    )
    write_json(destination / "dataset.json", data)
    return edits


def main():
    parser = argparse.ArgumentParser(
        description="Remove visually verified isolated border noise, preserving source mattes.\n\nUse only after inspecting RGB and alpha for the explicitly listed views.\nClipping of the main foreground component is never repaired by this operation."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--verified-background-border-views", type=int, nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(repair(args.source, args.destination, args.verified_background_border_views)))


if __name__ == "__main__":
    main()
