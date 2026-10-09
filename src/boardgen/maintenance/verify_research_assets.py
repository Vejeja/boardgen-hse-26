import argparse
import hashlib
import json

from boardgen.studio.environment import REPOSITORY as ROOT


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_index(record, snapshot, manifest):
    if "normalized_dataset_manifest_sha256" in record:
        normalized = hashlib.sha256(
            json.dumps(record["dataset"]["manifest"], ensure_ascii=False, indent=2).encode()
        ).hexdigest()
        require(normalized == record["normalized_dataset_manifest_sha256"], "Dataset record SHA")
        require(
            normalized == snapshot["normalized_dataset_manifest_sha256"], "Dataset snapshot SHA"
        )
    files = manifest["files"]
    dataset = f".boardgen/artifacts/{snapshot['dataset_id']}/"
    training = f".boardgen/artifacts/{snapshot['model_id']}/"
    require(snapshot["model_sha256"] == record["model_sha256"], "Wrong trained model")
    require(
        files[training + "run/weights/best.pt"]["sha256"] == record["model_sha256"],
        "Wrong best.pt in snapshot",
    )
    require(training + "run/weights/last.pt" in files, "Missing final checkpoint")
    require(
        files[dataset + "manifest.json"]["sha256"] == record["dataset_manifest_sha256"],
        "Wrong original dataset manifest",
    )
    require(snapshot["dataset_counts"] == record["dataset"]["counts"], "Wrong split counts")
    for split, count in record["dataset"]["counts"].items():
        for kind, suffix in [("images", ".png"), ("labels", ".txt")]:
            found = sum(
                name.startswith(dataset + f"{kind}/{split}/") and name.endswith(suffix)
                for name in files
            )
            require(found == count, f"Incomplete {kind}/{split}")
    figures = snapshot["selected_figures"]
    require(len(figures) == snapshot["class_count"] == 13, "Wrong selected class count")
    require([f["local_class_id"] for f in figures] == list(range(13)), "Wrong local class IDs")
    for figure in figures:
        name = f".boardgen/artifacts/{figure['asset_id']}/figure.glb"
        require(files[name]["sha256"] == figure["mesh_sha256"], f"Wrong GLB: {figure['object']}")
    require(
        any(
            item["sha256"] == record["initial_weights_sha256"]
            for name, item in files.items()
            if name.startswith("inputs/") and name.endswith("yolo11s.pt")
        ),
        "Missing exact initial COCO weights",
    )
    require("yolo11s.pt" in manifest["weight_names"], "Initial weights are not relocatable")


def check_comparison(root):
    package = root / "research/3d-quality"
    gallery = package / "gallery"
    catalog = json.loads((gallery / "comparison-catalog.json").read_text())
    assets = json.loads((gallery / "comparison-assets.json").read_text())
    recipes = json.loads((package / "comparison-recipes-20261009.json").read_text())
    require({row["object"] for row in catalog} == set(recipes), "Incomplete comparison collection")
    count = 0
    for row in catalog:
        data = json.loads((gallery / row["file"]).read_text())
        candidates = data["candidates"]
        require(
            {item["mesh_sha256"] for item in candidates}
            == {item["mesh_sha256"] for item in recipes[row["object"]]["candidates"]},
            f"Incomplete comparison variants: {row['object']}",
        )
        for item in candidates:
            require(assets[item["mesh"]] == item["mesh_sha256"], "Comparison model SHA")
            if item.get("coordinate_provenance"):
                require(item["coordinate_provenance"] in assets, "Missing coordinate provenance")
        for frame in data["frames"]:
            for key in ("photo", "mask", "contour"):
                require(frame[key] in assets, f"Missing comparison {key}")
            for key in ("projection", "view"):
                require(
                    len(frame[key]) == 4 and all(len(r) == 4 for r in frame[key]),
                    f"Invalid camera {key}",
                )
        require(len(candidates) == row["candidates"], "Catalog candidate count")
        require(len(data["frames"]) == row["frames"], "Catalog frame count")
        count += len(candidates)
    require(len(catalog) == 26 and count == 53, "Wrong comparison size")


def verify(root, full=False):
    bundle = root / "research/published-experiment"
    snapshot = json.loads((bundle / "snapshot.json").read_text())
    manifest = json.loads((bundle / "manifest.json").read_text())
    record = json.loads((root / "research/yolo/results.json").read_text())
    require(sha(bundle / "manifest.json") == snapshot["bundle_manifest_sha256"], "Manifest SHA")
    check_index(record, snapshot, manifest)
    weights = root / "research/yolo/weights/best.pt"
    if weights.exists():
        require(sha(weights) == record["model_sha256"], "Model SHA")
    for name, checksum in snapshot["additional_files"].items():
        require(sha(bundle / name) == checksum, f"Metadata SHA: {name}")
    for name, item in manifest["repository_files"].items():
        require(sha(root / name) == item["sha256"], f"Input SHA: {name}")
    package = root / "research/3d-quality"
    reconstruction = json.loads((package / "manifest.json").read_text())
    if full:
        require(
            all((bundle / part["name"]).is_file() for part in manifest["parts"]),
            "Local snapshot parts are missing; large assets are not included in Git",
        )
        for part in manifest["parts"]:
            path = bundle / part["name"]
            require(path.stat().st_size == part["bytes"], f"Part size: {part['name']}")
            require(sha(path) == part["sha256"], f"Part SHA: {part['name']}")
        for item in reconstruction["files"]:
            path = package / item["path"]
            require(path.is_file(), f"Local reconstruction file is missing: {item['path']}")
            require(path.stat().st_size == item["bytes"], f"3D size: {item['path']}")
            require(sha(path) == item["sha256"], f"3D SHA: {item['path']}")
        check_comparison(root)
    if full:
        from boardgen.studio.transfer import verify_bundle

        verify_bundle(bundle)
    weight_status = "OK" if weights.exists() else "absent"
    print(
        f"Published metadata: OK; local best.pt: {weight_status}; "
        f"{len(manifest['parts'])} snapshot parts, "
        f"{len(reconstruction['files'])} reconstruction files: "
        + ("fully verified" if full else "indexed; local binaries were not checked")
    )


def main():
    parser = argparse.ArgumentParser(
        description="Verify published metadata and available local weights; optionally verify local large assets."
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Require local large assets, verify their hashes and unpack the snapshot",
    )
    args = parser.parse_args()
    verify(ROOT, full=args.full)


if __name__ == "__main__":
    main()
