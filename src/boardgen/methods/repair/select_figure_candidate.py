from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from boardgen.evaluation.benchmark3d import file_hash, write_json
from boardgen.methods.repair.quality import score_candidate


def select(
    asset: Path,
    dataset: Path,
    candidate: Path,
    name: str,
    reason: str,
    max_regression: float = 0.02,
    accept_generated_approximation: bool = False,
    accept_paired_surface: bool = False,
):
    if not 0 <= max_regression <= 0.05:
        raise ValueError("Silhouette tolerance must be between zero and 0.05")
    for image in ["preview-photo.jpg", "neutral.jpg"]:
        if not (candidate / image).is_file():
            raise ValueError("Inspected candidate must have both full photo and neutral previews")
    meta = json.loads((asset / "asset.json").read_text())
    old_sha = file_hash(asset / "photo.glb")
    if old_sha != meta["files"]["photo.glb"]:
        raise ValueError("Current delivered mesh changed")
    metrics = candidate / "validation/score/metrics.json"
    score = (
        json.loads(metrics.read_text())
        if metrics.exists()
        else score_candidate(dataset, candidate, candidate / "validation")
    )
    if score["candidate_sha256"] != file_hash(candidate / "photo.glb"):
        raise ValueError("Candidate changed since validation")
    provenance = json.loads((candidate / "result.json").read_text())
    if provenance.get("photo_sha256") != score["candidate_sha256"]:
        raise ValueError("Candidate mesh differs from coordinate provenance")
    generation = provenance.get("generation_provenance")
    template = provenance.get("template_provenance")
    if template and generation:
        raise ValueError("Candidate cannot claim both generated and observed counterpart geometry")
    if template and (
        template["target_video_sha256"] != meta["source_video_sha256"]
        or template["source_object"] != f"obj_{int(meta['object'][4:]) + 1:02}"
    ):
        raise ValueError("Paired surface comes from a different figure")
    if generation and generation["source_video_sha256"] != meta["source_video_sha256"]:
        raise ValueError("Generated candidate comes from a different source video")
    old = meta["validation"][meta["selected"]]["validation"]
    new = score["summary"]["validation"]
    if accept_paired_surface and (
        not template or new["iou"] < 0.7 or new["foreground_recall"] < 0.8
    ):
        raise ValueError(
            "Explicit paired surface selection requires a matching counterpart and IoU >=0.7, recall >=0.8"
        )
    if accept_generated_approximation and (
        not generation or new["iou"] < 0.7 or new["foreground_recall"] < 0.85
    ):
        raise ValueError(
            "Explicit approximation selection requires a generated candidate with IoU >=0.7 and recall >=0.85"
        )
    if (
        new["iou"] < old["iou"] - max_regression
        or new["foreground_recall"] < old["foreground_recall"] - max_regression
    ) and not (accept_generated_approximation or accept_paired_surface):
        raise ValueError("Alternative regresses silhouette diagnostics; preserve current asset")
    archive = asset.parent / "archive" / f"asset-{old_sha[:16]}"
    archive.parent.mkdir(exist_ok=True)
    if archive.exists():
        raise ValueError("Selection archive already exists")
    asset.rename(archive)
    shutil.copytree(
        archive,
        asset,
        ignore=shutil.ignore_patterns(
            "visual-review.json",
            "neutral-preview*",
            "blender-*",
            "photo-*.png",
            "preview-photo.jpg",
            "preview-photo.json",
            "neutral.jpg",
        ),
    )
    for filename in [
        "photo.glb",
        "preview-photo.jpg",
        "preview-photo.json",
        "neutral.jpg",
        "neutral-preview.json",
        "neutral-preview-proxy.ply",
    ]:
        if (candidate / filename).exists():
            shutil.copy2(candidate / filename, asset / filename)
    meta.update(
        selected="refined",
        status="candidate_needs_visual_review",
        files={p.name: file_hash(p) for p in asset.glob("*.glb")},
    )
    meta["validation"]["refined"] = score["summary"]
    meta["refinement"]["selected"] = name
    meta["refinement"]["scores"][name] = score["summary"]
    meta["refinement"]["candidate_provenance"] = str((candidate / "result.json").resolve())
    meta["refinement"]["manual_selection_reason"] = reason
    meta["refinement"]["allowed_silhouette_regression"] = max_regression
    meta["refinement"]["explicit_generated_approximation_selection"] = (
        accept_generated_approximation
    )
    meta["refinement"]["explicit_paired_surface_selection"] = accept_paired_surface
    meta["refinement"]["silhouette_change"] = {
        key: new[key] - old[key] for key in ["iou", "foreground_recall"]
    }
    meta["refinement"]["previous_asset"] = str(archive.resolve())
    if provenance.get("appearance"):
        meta["appearance"] = provenance["appearance"]
    else:
        meta.pop("appearance", None)
    if "generation_provenance" in provenance:
        generation = provenance["generation_provenance"]
        meta["selected_method"] = "hunyuan_mv"
        meta["settings"]["training_role"] = (
            "Spirula comparison baseline; selected geometry comes from the generated MV candidate described below"
        )
        meta["settings"]["selected_candidate"] = {
            "method": generation["method"],
            "parameters": generation["parameters"],
            "weights": generation["weights"],
            "inputs": generation["inputs"],
            "conditioning_photographs": len(generation["inputs"]),
            "port_commit": generation["port_commit"],
        }
        meta["refinement"]["assessment"] = provenance["assessment"]
    elif template:
        meta["selected_method"] = "paired_observed_surface"
        meta["settings"]["training_role"] = (
            "Target Spirula comparison baseline; selected surface reconstructed from the red counterpart"
        )
        meta["settings"]["selected_candidate"] = template
        meta["refinement"]["assessment"] = provenance["assessment"]
    write_json(asset / "asset.json", meta)
    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Select an inspected alternative without losing the previous delivered asset."
    )
    parser.add_argument("asset", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--max-regression", type=float, default=0.02)
    parser.add_argument(
        "--accept-paired-surface",
        action="store_true",
        help="explicitly select an observed red counterpart surface with target photo texture; records the shared-sculpt assumption and diagnostics",
    )
    parser.add_argument(
        "--accept-generated-approximation",
        action="store_true",
        help="explicit visual selection of a complete generated shape despite worse IoU; retains diagnostics and is reported as an approximation",
    )
    args = parser.parse_args()
    meta = select(
        args.asset.resolve(),
        args.dataset.resolve(),
        args.candidate.resolve(),
        args.name,
        args.reason,
        args.max_regression,
        args.accept_generated_approximation,
        args.accept_paired_surface,
    )
    print(
        meta["object"], meta["refinement"]["selected"], meta["validation"]["refined"]["validation"]
    )


if __name__ == "__main__":
    main()
