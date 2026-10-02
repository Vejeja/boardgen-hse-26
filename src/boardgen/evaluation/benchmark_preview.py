from __future__ import annotations

import json
import struct
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def read_colmap(dataset: Path) -> list[tuple[str, np.ndarray, int, int]]:
    sparse = dataset / "sparse/0"
    cameras = {}
    with (sparse / "cameras.bin").open("rb") as stream:
        count = struct.unpack("<Q", stream.read(8))[0]
        for _ in range(count):
            cid, kind, width, height = struct.unpack("<iiQQ", stream.read(24))
            if kind == 0:
                focal, cx, cy = struct.unpack("<ddd", stream.read(24))
                fx = fy = focal
            elif kind == 1:
                fx, fy, cx, cy = struct.unpack("<dddd", stream.read(32))
            else:
                raise ValueError("preview expects SIMPLE_PINHOLE or PINHOLE without distortion")
            cameras[cid] = (np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]), width, height)
    views = []
    with (sparse / "images.bin").open("rb") as stream:
        count = struct.unpack("<Q", stream.read(8))[0]
        for _ in range(count):
            _, qw, qx, qy, qz, tx, ty, tz, cid = struct.unpack("<idddddddi", stream.read(64))
            name = bytearray()
            while True:
                byte = stream.read(1)
                if not byte:
                    raise ValueError("truncated COLMAP image name")
                if byte == b"\0":
                    break
                name.extend(byte)
            points = struct.unpack("<Q", stream.read(8))[0]
            stream.seek(points * 24, 1)
            k, width, height = cameras[cid]
            r = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            projection = k @ np.column_stack([r, [tx, ty, tz]])
            views.append((name.decode("utf8"), projection, width, height))
    return sorted(views, key=lambda view: view[0])






