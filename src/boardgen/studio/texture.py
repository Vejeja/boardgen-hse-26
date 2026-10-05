import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from boardgen.camera.camera_audit import read_model
from boardgen.evaluation.benchmark3d import run_command
from boardgen.methods.mv.texture_mv import extend_foreground, restore_uv

from .environment import TOOLS
from .storage import json_write


def bake_reconstruction(root: Path, source: Path, output: Path, texture_size: int):
    data = json.loads((root / "dataset.json").read_text())
    cameras, images, _ = read_model(root / "train/sparse/0")
    transform = np.array(json.loads((source.parent / "asset.json").read_text())["transform"])
    mesh = trimesh.load(source, force="mesh")
    mesh.apply_transform(np.linalg.inv(transform))
    output.mkdir()
    mesh.export(output / "mesh.ply")
    folder = output / "colmap/images"
    folder.mkdir(parents=True)
    sparse = output / "colmap/sparse"
    sparse.mkdir()
    rows = []
    for camera_id, c in cameras.items():
        rows.append(
            f"{camera_id} PINHOLE {c['width']} {c['height']} {c['fx']} {c['fy']} {c['cx']} {c['cy']}\n"
        )
    (sparse / "cameras.txt").write_text("".join(rows))
    rows = []
    for image_id, image in enumerate(images, 1):
        name = image["name"]
        index = int(name[5:9])
        if index not in data["fit"]:
            raise ValueError("Кадр проверки не должен участвовать в текстурировании")
        rgb = cv2.imread(str(root / "reference/images" / name))
        mask = cv2.imread(str(root / "reference/masks" / name), 0) >= 128
        cv2.imwrite(str(folder / name), extend_foreground(rgb, mask))
        cv2.imwrite(str(folder / f"{Path(name).stem}.mask.png"), mask.astype("uint8") * 255)
        q = Rotation.from_matrix(image["r"]).as_quat()
        t = image["t"]
        rows.append(
            f"{image_id} {q[3]} {q[0]} {q[1]} {q[2]} {t[0]} {t[1]} {t[2]} {image['cid']} {name}\n\n"
        )
    (sparse / "images.txt").write_text("".join(rows))
    (sparse / "points3D.txt").write_text("# Frozen reconstruction geometry\n")
    for name, binary, args, expected in [
        (
            "import",
            "InterfaceCOLMAP",
            ["-i", output / "colmap", "-o", output / "scene.mvs", "--image-folder", folder],
            ["scene.mvs"],
        ),
        (
            "texture",
            "TextureMesh",
            [
                "-i",
                output / "scene.mvs",
                "-m",
                output / "mesh.ply",
                "-o",
                output / "textured.obj",
                "--export-type",
                "obj",
                "--decimate",
                1,
                "--close-holes",
                0,
                "--resolution-level",
                0,
                "--max-texture-size",
                texture_size,
                "--virtual-face-images",
                3,
                "--cost-smoothness-ratio",
                0.01,
                "--global-seam-leveling",
                0,
                "--local-seam-leveling",
                0,
                "--sharpness-weight",
                0,
                "--ignore-mask-label",
                0,
            ],
            ["textured.obj", "textured.mtl"],
        ),
    ]:
        row = run_command(
            output / name,
            [
                str(TOOLS / "openmvs" / binary),
                "-w",
                str(output),
                "--max-threads",
                "4",
                *map(str, args),
            ],
            [str(output / p) for p in expected],
            timeout=300,
        )
        if row["status"] != "success":
            raise RuntimeError(f"Ошибка фототекстуры: {output / name / 'process.log'}")
    geometry = restore_uv(mesh, output / "textured.obj", output / "photo.glb")
    model = trimesh.load(output / "photo.glb", force="mesh", process=False)
    model.apply_transform(transform)
    model.export(output / "photo.glb")
    json_write(
        output / "result.json",
        {"geometry": geometry, "fit_only": True, "transform": transform.tolist()},
    )
    return output / "photo.glb"
