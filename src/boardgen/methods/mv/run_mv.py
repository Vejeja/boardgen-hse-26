import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np
import trimesh

from boardgen.evaluation.benchmark3d import file_hash, write_json
from boardgen.methods.mv.conditioning import conditioning_image
from boardgen.methods.mv.mv_adapter import pipeline_class

PORT_COMMIT = "292331f4d26ddb80b9dcea6bcb5629ff82f12b82"


def main():
    parser = argparse.ArgumentParser(
        description="Bounded genuine MV-checkpoint inference, using only frozen fit photographs."
    )
    parser.add_argument("root", type=Path)
    parser.add_argument("views", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--weights", type=Path, default=Path("tmp/benchmark_tools/hunyuan-mv-turbo")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).with_name("mv-model.json"),
    )
    parser.add_argument("--front-only", action="store_true")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--quantize", type=int, choices=[4, 8], default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--save-grid", action="store_true")
    parser.add_argument("--conditioning-alpha-threshold", type=int, default=128)
    parser.add_argument("--foreground-gain", type=float, default=1.0)
    parser.add_argument("--seconds", type=int, default=840)
    args = parser.parse_args()
    dataset = json.loads((args.root / "dataset.json").read_text())
    selection = json.loads(args.views.read_text())
    weights = json.loads(args.manifest.read_text())
    if selection["root"] != args.root.name:
        raise ValueError("view selection belongs to another dataset")
    views = selection["views"]
    if args.front_only:
        views = [v for v in views if v["tag"] == "front"]
    if any(v["view"] not in dataset["fit"] for v in views):
        raise ValueError("held-out input forbidden")
    if len({v["view"] for v in views}) != len(views):
        raise ValueError("duplicate input photograph")
    if args.out.exists():
        raise ValueError("fresh output required")
    if file_hash(args.weights / "model.fp16.safetensors") != weights["sha256"]:
        raise ValueError("checkpoint checksum mismatch")
    args.out.mkdir(parents=True)
    images = {v["tag"]: args.root / "train/images" / f"view_{v['view']:04d}.png" for v in views}
    conditioning = args.out / "conditioning"
    conditioning.mkdir()
    prepared = {}
    for tag, source in images.items():
        target = conditioning / f"{tag}.png"
        conditioning_image(source, args.conditioning_alpha_threshold, args.foreground_gain).save(
            target
        )
        prepared[tag] = target
    record = {
        "method": "Hunyuan3D-2mv-turbo; MLX MV conditioner adaptation",
        "port_commit": PORT_COMMIT,
        "weights": weights,
        "config_sha256": file_hash(args.weights / "config.yaml"),
        "source_video_sha256": dataset["source_video_sha256"],
        "inputs": [{**v, "sha256": file_hash(images[v["tag"]])} for v in views],
        "selection": selection["selection"],
        "parameters": {
            "steps": args.steps,
            "resolution": args.resolution,
            "quantize": args.quantize,
            "seed": args.seed,
            "guidance": 5.0,
            "query_chunk": 4096,
            "save_grid": args.save_grid,
            "front_only": args.front_only,
            "conditioning_alpha_threshold": args.conditioning_alpha_threshold,
            "foreground_gain": args.foreground_gain,
        },
        "conditioning_inputs": {
            tag: {"path": str(path), "sha256": file_hash(path)} for tag, path in prepared.items()
        },
        "timing_scope": "Model load, generation and full mesh export; reuses previously prepared masks. One-time checkpoint download and dataset preparation excluded.",
        "status": "running",
    }
    write_json(args.out / "result.json", record)

    def timeout(*_):
        raise TimeoutError("per-figure calculation budget exceeded")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.seconds)
    start = time.perf_counter()
    try:
        import mlx.core as mx

        mx.set_cache_limit(256 << 20)
        mx.set_memory_limit(8 << 30)
        pipeline = pipeline_class().from_pretrained(
            args.weights, dtype=mx.float16, quantize=args.quantize
        )
        if args.save_grid:
            original_extract = pipeline._grid_to_mesh

            def save_and_extract(grid, bbox_min, bbox_max, grid_size, mc_level):
                np.savez_compressed(
                    args.out / "decoded-grid.npz",
                    grid=grid,
                    bbox_min=bbox_min,
                    bbox_max=bbox_max,
                    grid_size=grid_size,
                    mc_level=mc_level,
                )
                return original_extract(grid, bbox_min, bbox_max, grid_size, mc_level)

            pipeline._grid_to_mesh = save_and_extract
        loaded = time.perf_counter()
        mesh = pipeline.generate(
            prepared,
            num_inference_steps=args.steps,
            guidance_scale=5.0,
            octree_resolution=args.resolution,
            num_chunks=4096,
            seed=args.seed,
            octree_decode=True,
        )
        generated = time.perf_counter()
        if not np.isfinite(mesh.vertices).all() or len(mesh.faces) < 500:
            raise ValueError("invalid generated mesh")
        mesh.export(args.out / "raw.ply")
        offset = [
            -float(mesh.bounds[:, 0].mean()),
            -float(mesh.bounds[0, 1]),
            -float(mesh.bounds[:, 2].mean()),
        ]
        scale = 1 / np.ptp(mesh.vertices[:, 1])
        mesh.apply_translation(offset)
        mesh.apply_scale(scale)
        mesh.visual = trimesh.visual.TextureVisuals(
            uv=np.zeros((len(mesh.vertices), 2)),
            material=trimesh.visual.material.PBRMaterial(
                baseColorFactor=[160, 160, 160, 255], metallicFactor=0, roughnessFactor=0.7
            ),
        )
        mesh.export(args.out / "neutral.glb")
        record.update(
            status="success",
            model_load_seconds=loaded - start,
            generation_seconds=generated - loaded,
            peak_mlx_bytes=mx.get_peak_memory(),
            faces=len(mesh.faces),
            watertight=bool(mesh.is_watertight),
            mesh_sha256=file_hash(args.out / "neutral.glb"),
            transform={"translation_before_scale": offset, "uniform_scale": float(scale)},
        )
    except Exception as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        signal.alarm(0)
        record["total_seconds"] = time.perf_counter() - start
        write_json(args.out / "result.json", record)


if __name__ == "__main__":
    main()
