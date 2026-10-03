import csv
from pathlib import Path

from boardgen.studio.environment import REPOSITORY as PROJECT


def figure_names(source: Path | None = None):
    source = source or PROJECT / "metadata/video_index.csv"
    names = {}
    slugs = set()
    with source.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            obj = Path(row["video"]).stem
            slug = row["yolo_class"]
            if obj in names or slug in slugs:
                raise ValueError(f"Duplicate video or class in {source}: {row['video']}")
            if not row["ru_name"] or not slug or Path(slug).name != slug:
                raise ValueError(f"Missing name or invalid class in {source}: {row['video']}")
            names[obj] = {
                "name": row["ru_name"],
                "slug": slug,
                "basis": "metadata/video_index.csv",
                "video": row["video"],
                "class_id": int(row["class_id"]),
            }
            slugs.add(slug)
    return names
