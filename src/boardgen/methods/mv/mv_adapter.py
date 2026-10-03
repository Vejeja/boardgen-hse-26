import os
import sys
from pathlib import Path

import numpy as np

TAGS = {"front": 0, "left": 1, "back": 2, "right": 3}
PORT = (
    Path(os.environ.get("BOARDGEN_TOOLS_DIR", "tmp/benchmark_tools"))
    / "hunyuan3d-swift/python/shape"
)


def ordered_views(images):
    if not images or set(images) - TAGS.keys():
        raise ValueError("expected nonempty subset of front/left/back/right")
    return sorted(images.items(), key=lambda pair: TAGS[pair[0]])


def view_embeddings(tags, hidden):
    if hidden % 2:
        raise ValueError("view embedding dimension must be even")
    positions = np.array([TAGS[tag] for tag in tags], dtype=np.float32)
    omega = 1 / 10000 ** (np.arange(hidden // 2, dtype=np.float64) / (hidden // 2))
    phase = positions[:, None] * omega[None, :]
    return np.concatenate([np.sin(phase), np.cos(phase)], axis=1).astype(np.float32)


def pipeline_class():
    sys.path.insert(0, str(PORT.resolve()))
    import mlx.core as mx
    from hy3dmlx.pipeline import Hunyuan3DShapePipeline
    from hy3dmlx.preprocess import dino_transform, load_image

    class MultiviewPipeline(Hunyuan3DShapePipeline):
        @classmethod
        def from_pretrained(cls, model_dir, dtype=mx.float16, quantize=8):
            from hy3dmlx import convert
            from hy3dmlx.models.shape_vae import ShapeVAE
            from safetensors import safe_open

            cfg = convert.load_config(str(model_dir))
            blocks = {}
            with safe_open(str(Path(model_dir) / "model.fp16.safetensors"), framework="numpy") as f:
                for name, prefix in [
                    ("dino", convert.COND_PREFIX),
                    ("dit", "model."),
                    ("vae", "vae."),
                ]:
                    items = {}
                    for key in f.keys():
                        if key.startswith(prefix):
                            value = f.get_tensor(key)
                            local = key[len(prefix) :]
                            if (
                                local.endswith("patch_embeddings.projection.weight")
                                and value.ndim == 4
                            ):
                                value = np.transpose(value, (0, 2, 3, 1))
                            items[local] = mx.array(value).astype(dtype)
                    if name == "dino":
                        module = convert._build_dino(cfg, items)
                    elif name == "dit":
                        module = convert._build_dit(cfg)
                    else:
                        module = ShapeVAE(**cfg["vae"]["params"])
                    counts = convert._assign(module, items.items(), name)
                    mx.eval(module.parameters())
                    del items, value
                    if quantize and name != "vae":
                        skip = (
                            convert._QUANT_SKIP_DINO if name == "dino" else convert._QUANT_SKIP_DIT
                        )
                        convert._quantize(module, skip, quantize)
                        mx.eval(module.parameters())
                    blocks[name] = module
                    mx.clear_cache()
                    print(
                        f"loaded {name}: {counts}, active {mx.get_active_memory() / 1e9:.2f}GB",
                        flush=True,
                    )
            return cls(blocks["dino"], blocks["dit"], blocks["vae"], cfg, dtype)

        def encode_image(self, image_paths, border_ratio=0.15):
            spec = self.cfg["conditioner"]["params"]["main_image_encoder"]
            if spec["type"] != "DinoImageEncoderMV":
                raise ValueError("requires an actual pretrained MV checkpoint")
            ordered = ordered_views(image_paths)
            embedding = view_embeddings(
                [tag for tag, _ in ordered], spec["kwargs"]["config"]["hidden_size"]
            )
            features = []
            for i, (_, path) in enumerate(ordered):
                image = load_image(str(path), size=512, border_ratio=border_ratio)
                pixels = dino_transform(image[0]).astype(self.dtype)
                token = self.dino(pixels)
                token = token + mx.array(embedding[i])[None, None, :].astype(token.dtype)
                mx.eval(token)
                features.append(token)
                mx.clear_cache()
            conditional = mx.concatenate(features, axis=1)
            unconditional = mx.zeros_like(conditional)
            combined = mx.concatenate([conditional, unconditional], axis=0)
            mx.eval(combined)
            print(f"MV conditioning: {[tag for tag, _ in ordered]}, {combined.shape}", flush=True)
            return combined

    return MultiviewPipeline
