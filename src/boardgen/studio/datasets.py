from __future__ import annotations

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .storage import json_write, sha

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def labels(path: Path, class_count: int, allow_empty=False):
    if not path.exists():
        if allow_empty:
            return []
        raise ValueError(
            f"Нет разметки: {path}. Пустые изображения нужно явно подтвердить как фон."
        )
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_no}: требуется bbox YOLO detect из 5 чисел")
        values = [float(v) for v in fields]
        cls, x, y, w, h = values
        if not all(np.isfinite(values)) or cls != int(cls) or not 0 <= cls < class_count:
            raise ValueError(f"{path}:{line_no}: неизвестный класс или неверные числа")
        if (
            not 0 < w <= 1
            or not 0 < h <= 1
            or not 0 <= x <= 1
            or not 0 <= y <= 1
            or x - w / 2 < -1e-5
            or x + w / 2 > 1.00001
            or y - h / 2 < -1e-5
            or y + h / 2 > 1.00001
        ):
            raise ValueError(f"{path}:{line_no}: рамка выходит за границы изображения")
        rows.append([int(cls), x, y, w, h])
    return rows


def image_hash(path):
    with Image.open(path) as im:
        pixels = np.array(im.convert("L").resize((9, 8)))
    bits = pixels[:, 1:] > pixels[:, :-1]
    return int.from_bytes(np.packbits(bits).tobytes(), "big")


def split_groups(records, fraction, seed):
    fixed = {}
    for row in records:
        if row.get("fixed_split"):
            previous = fixed.setdefault(row["group"], row["fixed_split"])
            if previous != row["fixed_split"]:
                raise ValueError("Одна группа назначена разным разделам")
    groups = sorted({r["group"] for r in records} - set(fixed))
    random.Random(seed).shuffle(groups)
    existing = {g: s for g, s in fixed.items() if s != "test"}
    total = len(groups) + len(existing)
    if total < 2:
        raise ValueError("Для train и val нужны как минимум две независимые группы изображений")
    fixed_val = sum(v == "val" for v in existing.values())
    fixed_train = sum(v == "train" for v in existing.values())
    n_val = round(total * fraction) - fixed_val
    minimum = 0 if fixed_val else 1
    maximum = len(groups) if fixed_train else len(groups) - 1
    n_val = max(minimum, min(maximum, n_val))
    if n_val > len(groups) or n_val < 0:
        raise ValueError("В готовом датасете нужны отдельные train и val")
    val = set(groups[:n_val])
    return {**existing, **{g: "val" if g in val else "train" for g in groups}}


def import_source(store, source, names):
    config_path = store.resolve(source.path, exists=True)
    document = yaml.safe_load(config_path.read_text())
    if not isinstance(document, dict):
        raise ValueError("Некорректный data.yaml")
    original_names = document.get("names", [])
    if isinstance(original_names, dict):
        if sorted(int(k) for k in original_names) != list(range(len(original_names))):
            raise ValueError("ID классов data.yaml должны идти от 0 без пропусков")
        original_names = [
            original_names.get(i, original_names.get(str(i))) for i in range(len(original_names))
        ]
    if (
        not isinstance(original_names, list)
        or not original_names
        or not all(isinstance(n, str) and n.strip() for n in original_names)
        or len(set(original_names)) != len(original_names)
    ):
        raise ValueError("В data.yaml отсутствуют уникальные имена классов")
    if not names:
        names.extend(original_names)
    if set(original_names) - set(names):
        raise ValueError(
            f"Классы data.yaml не совпадают с проектом: {set(original_names) - set(names)}"
        )
    mapping = {i: names.index(n) for i, n in enumerate(original_names)}
    base = store.resolve(str(config_path.parent / document.get("path", ".")), exists=True)
    group_map = {}
    if source.groups_csv:
        with store.resolve(source.groups_csv, exists=True).open() as stream:
            for row in csv.DictReader(stream):
                if not row.get("image") or not row.get("group"):
                    raise ValueError("CSV групп должен содержать столбцы image,group")
                name = Path(row["image"]).as_posix()
                if name in group_map and group_map[name] != row["group"]:
                    raise ValueError(f"CSV задаёт разные группы для {name}")
                group_map[name] = row["group"]
    records = []
    splits = (
        ["test"]
        if source.role == "test_only" and document.get("test")
        else (["val"] if source.role == "test_only" else ["train", "val"])
    )
    for split in splits:
        entries = document.get(split, [])
        entries = entries if isinstance(entries, list) else [entries]
        images = []
        for entry in entries:
            entry_path = store.resolve(str(base / entry), exists=True)
            if entry_path.is_dir():
                images.extend(
                    p for p in sorted(entry_path.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES
                )
            elif entry_path.suffix.lower() == ".txt":
                images.extend(
                    store.resolve(str(entry_path.parent / line.strip()), exists=True)
                    for line in entry_path.read_text().splitlines()
                    if line.strip()
                )
            else:
                images.append(entry_path)
        for image in images:
            parts = list(image.parts)
            if "images" not in parts:
                raise ValueError("Импорт YOLO ожидает параллельные папки images/ и labels/")
            parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
            label = Path(*parts).with_suffix(".txt")
            store.resolve(str(label))
            store.resolve(str(image), exists=True)
            rows = labels(label, len(original_names), source.negative_labels_confirmed)
            for row in rows:
                row[0] = mapping[row[0]]
            relative = image.relative_to(base).as_posix()
            if source.groups_csv and relative not in group_map:
                raise ValueError(f"В CSV нет группы для {relative}")

            group = group_map.get(relative, f"existing-{split}")
            records.append(
                {
                    "image": str(image),
                    "labels": rows,
                    "group": f"real:capture:{group}"
                    if source.groups_csv
                    else f"real:{base}:{group}",
                    "origin": str(config_path),
                    "fixed_split": "test"
                    if source.role == "test_only"
                    else (split if not source.groups_csv else None),
                }
            )
    if not records:
        raise ValueError(f"В источнике нет изображений: {config_path}")
    return records


def audit_backgrounds(records, policy):
    if policy == "none":
        return {"policy": policy}
    sources = {key: {} for key in ("board_sha256", "skybox_sha256")}
    for record in records:
        if record.get("origin") != "synthetic":
            continue
        split = record.get("split") or record.get("fixed_split")
        if split not in ("train", "val", "test"):
            raise ValueError("Проверка фонов требует фиксированный раздел сцены")
        role = "development" if policy == "test_only" and split != "test" else split
        for key, owners in sources.items():
            value = record.get(key.replace("_sha256", "_source_id")) or record.get(key)
            if not value and key == "board_sha256" and record.get("scene_type") == "platform":
                continue
            if not value:
                raise ValueError(f"Нет происхождения фона: {key}")
            if owners.setdefault(value, role) != role:
                raise ValueError(f"Источник фона пересекает разделы: {key}")
    return {"policy": policy, "unique_sources": {k: len(v) for k, v in sources.items()}}


def build(store, project, destination: Path, synthetic_records):
    names = [f.name for f in project.figures]
    records = list(synthetic_records)
    warnings = []
    for source in project.dataset.real_sources:
        records.extend(import_source(store, source, names))
        if not source.groups_csv:
            warnings.append(
                f"{source.path}: группы взяты из исходных папок train/val; для разделения по съёмкам подключите CSV"
            )
    if not records:
        raise ValueError("Нет изображений для датасета")
    background_audit = audit_backgrounds(records, project.dataset.background_policy)
    test_only = all(r.get("fixed_split") == "test" for r in records)
    assignments = (
        {}
        if test_only
        else split_groups(records, project.dataset.val_fraction, project.dataset.seed)
    )
    groups_test = {r["group"] for r in records if r.get("fixed_split") == "test"}
    if groups_test & set(assignments):
        raise ValueError("Одна группа попала в обучение и тест")
    checksums = {}
    near = []
    buckets = defaultdict(list)
    counts = defaultdict(int)
    published = []
    for i, record in enumerate(records):
        image = Path(record["image"])
        checksum = sha(image)
        if checksum in checksums:
            raise ValueError(f"Дублирующиеся изображения: {image} и {checksums[checksum]}")
        checksums[checksum] = str(image)
        h = image_hash(image)
        split = record.get("fixed_split") or assignments[record["group"]]
        candidates = set()
        for block in range(4):
            candidates.update(buckets[(block, (h >> (block * 16)) & 65535)])
        for previous in candidates:
            prev = published[previous]
            if prev["split"] != split and (h ^ prev["perceptual_hash"]).bit_count() <= 3:
                near.append(
                    {
                        "a": prev["name"],
                        "b": f"image_{i:07d}",
                        "distance": (h ^ prev["perceptual_hash"]).bit_count(),
                    }
                )
        for block in range(4):
            buckets[(block, (h >> (block * 16)) & 65535)].append(i)
        name = f"image_{i:07d}"
        output = destination / "images" / split / f"{name}{image.suffix.lower()}"
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, output)
        label_path = destination / "labels" / split / f"{name}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(
            "".join(f"{r[0]} {' '.join(f'{v:.6f}' for v in r[1:])}\n" for r in record["labels"])
        )
        published.append(
            {
                **record,
                "image": str(output.relative_to(destination)),
                "name": name,
                "split": split,
                "sha256": checksum,
                "perceptual_hash": h,
            }
        )
        counts[split] += 1
    if near and project.dataset.near_duplicates == "error":
        raise ValueError(f"Найдено {len(near)} похожих пар между разделами")
    data = {
        "path": str(destination),
        "train": "images/train" if counts["train"] else None,
        "val": "images/val" if counts["val"] else None,
        "names": dict(enumerate(names)),
    }
    if counts["test"]:
        data["test"] = "images/test"
    selection_map = store.root / "class-map.json"
    if selection_map.is_file():
        import json

        source_map = json.loads(selection_map.read_text())
        by_object = {row["object"]: row for row in source_map["figures"]}
        mapped = []
        for local_id, figure in enumerate(project.figures):
            if figure.id not in by_object:
                raise ValueError(f"Нет исходного класса для выбранной фигуры: {figure.name}")
            row = {
                key: by_object[figure.id][key]
                for key in ("object", "source_class_id", "yolo_class")
            }
            row.update(local_class_id=local_id, name=figure.name)
            asset_id = project.scenes.asset_versions.get(figure.id)
            if asset_id and synthetic_records:
                item = store.artifact(asset_id)
                row.update(asset_id=asset_id, mesh_sha256=item["manifest"]["files"]["figure.glb"])
            mapped.append(row)
        json_write(
            destination / "class-map.json",
            {
                "schema_version": 1,
                "name_source_sha256": source_map["name_source_sha256"],
                "figures": mapped,
            },
        )
    (destination / "data.yaml").write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
    )
    report = {
        "schema": "boardgen.dataset.v1",
        "background_audit": background_audit,
        "synthetic_annotation_encodings": sorted(
            {
                r.get("scene", {}).get("flags", {}).get("id_encoding", "scalar")
                for r in records
                if r.get("origin") == "synthetic"
            }
        ),
        "counts": dict(counts),
        "names": names,
        "groups": assignments,
        "test_groups": sorted(groups_test),
        "exact_duplicates": 0,
        "near_duplicates": near[:1000],
        "near_duplicate_count": len(near),
        "near_duplicate_method": "64-bit dHash, Hamming ≤3; heuristic, not a semantic guarantee",
        "warnings": warnings,
        "records": published,
    }
    json_write(destination / "dataset.json", report)
    return report
