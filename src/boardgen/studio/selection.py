from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from boardgen.common.figure_names import figure_names

from . import environment
from .models import Figure, Model, Project
from .preparation import asset_metadata
from .storage import Store, digest, json_write, sha


class Choice(Model):
    object: str = Field(pattern=r"^obj_\d{2}$")
    id: str = Field(min_length=1, max_length=120)
    mesh_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    name: str | None = None
    label: str | None = None
    chosen_at: str | None = None


class Selection(Model):
    schema_version: Literal[1]
    choices: list[Choice] = Field(min_length=1, max_length=26)

    @model_validator(mode="after")
    def unique_figures(self):
        objects = [c.object for c in self.choices]
        if len(set(objects)) != len(objects):
            raise ValueError("Выберите только одну модель для каждой фигурки")
        return self


def available_packages(repository: Path) -> list[Path]:
    root = (repository / "out").resolve()
    packages = [
        p.parent.parent.resolve()
        for p in root.glob("*/gallery/comparison-catalog.json")
        if p.resolve().is_relative_to(root)
        and (p.parent.parent / "completion-report.json").is_file()
    ]
    preserved = repository / "research/3d-quality"
    if (
        preserved.resolve().is_relative_to(repository.resolve())
        and (preserved / "gallery/comparison-catalog.json").is_file()
        and (preserved / "completion-report.json").is_file()
    ):
        packages.append(preserved.resolve())
    return sorted(
        packages, key=lambda p: (p / "completion-report.json").stat().st_mtime, reverse=True
    )


def validate_selection(selection, package, repository):
    names = figure_names(repository / "metadata/video_index.csv")
    catalog = json.loads((package / "gallery/comparison-catalog.json").read_text())
    catalog_by_object = {entry["object"]: entry for entry in catalog}
    records = []
    metadata = {}
    provenance_files = {}
    for choice in sorted(
        selection.choices, key=lambda c: names.get(c.object, {}).get("class_id", -1)
    ):
        if choice.object not in names or choice.object not in catalog_by_object:
            raise ValueError(f"Неизвестная фигурка: {choice.object}")
        reference = (package / "gallery" / catalog_by_object[choice.object]["file"]).resolve()
        if not reference.is_relative_to(package):
            raise ValueError("Каталог сравнения выходит за пределы набора")
        data = json.loads(reference.read_text())
        entry = next((c for c in data["candidates"] if c["id"] == choice.id), None)
        if (
            data["object"] != choice.object
            or data["source_video_sha256"] != choice.source_video_sha256
            or not entry
            or entry["mesh_sha256"] != choice.mesh_sha256
        ):
            raise ValueError(f"Выбор устарел или относится к другой модели: {choice.object}")
        mesh = (package / "gallery" / entry["mesh"]).resolve()
        if not mesh.is_relative_to(package) or mesh.suffix.lower() != ".glb":
            raise ValueError("Модель должна быть GLB внутри выбранного набора")
        name = names[choice.object]
        video = repository / "research/3d-quality/videos" / name["video"]
        if sha(mesh) != choice.mesh_sha256 or sha(video) != choice.source_video_sha256:
            raise ValueError(f"Контрольная сумма модели или видео изменилась: {choice.object}")
        if entry.get("coordinate_provenance"):
            provenance = (package / "gallery" / entry["coordinate_provenance"]).resolve()
            assets = json.loads((package / "gallery/comparison-assets.json").read_text())
            if not provenance.is_relative_to(package) or sha(provenance) != assets.get(
                entry["coordinate_provenance"]
            ):
                raise ValueError(f"Запись происхождения модели изменилась: {choice.object}")
            provenance_files[choice.object] = provenance
        figure = Figure(id=choice.object, name=name["name"], video=str(video))

        meta = asset_metadata(
            mesh,
            figure,
            Project().prepare.resolved(figure.id),
            {"id": None, "manifest": {"metadata": {"source_sha256": choice.source_video_sha256}}},
            {},
        )
        meta.update(
            origin="gallery-selection",
            recipe=None,
            appearance="native"
            if entry.get("appearance_method") == "uniform_observed_plastic_colour"
            else "photo",
            appearance_method=entry.get("appearance_method"),
            method=entry.get("method"),
            original_sha256=choice.mesh_sha256,
            research_title=entry["label"],
            research_note=entry["note"],
            quality_state="manually_selected_candidate",
            selection=choice.model_dump(),
        )
        metadata[choice.object] = meta
        records.append(
            {
                "local_class_id": len(records),
                "source_class_id": name["class_id"],
                "object": choice.object,
                "name": name["name"],
                "yolo_class": name["slug"],
                "video": str(video),
                "source_video_sha256": choice.source_video_sha256,
                "candidate_id": choice.id,
                "candidate_label": entry["label"],
                "mesh": str(mesh),
                "mesh_sha256": choice.mesh_sha256,
            }
        )
    return records, metadata, provenance_files


def import_selection(
    content: dict,
    package: Path,
    destination: Path,
    repository: Path | None = None,
    count: int = 1000,
) -> dict:
    repository = (repository or environment.REPOSITORY).resolve()
    package, destination = package.resolve(), destination.resolve()
    selection = Selection.model_validate(content)
    if destination.exists():
        raise ValueError("Для выбора нужен новый каталог проекта; прежние результаты сохраняются")
    records, metadata, provenance_files = validate_selection(selection, package, repository)
    project = Project(
        name="YOLO: выбранные фигурки",
        figures=[Figure(id=r["object"], name=r["name"], video=r["video"]) for r in records],
        scenes={
            "count": count,
            "objects_per_scene": [1, 8],
            "negative_fraction": 0.12,
            "min_side_px": 32,
            "camera": {"elevation_deg": [25, 68], "azimuth_deg": [-180, 180]},
            "boards": [
                {
                    "image": str(repository / "data/board/board1.png"),
                    "width_m": 0.5,
                    "height_m": 0.5,
                }
            ]
            if (repository / "data/board/board1.png").exists()
            else [],
            "skyboxes": [
                str(repository / "data/skybox" / name)
                for name in ["room.jpg", "room2.jpg", "room3.jpeg"]
                if (repository / "data/skybox" / name).exists()
            ],
        },
        training={"optimizer": "AdamW", "lr0": 0.001, "batch": 8, "workers": 0, "mosaic": 0.2},
    )

    store = Store(destination)
    store.grant(str(repository))
    for row in records:
        folder = destination / ".boardgen/import" / row["object"]
        folder.mkdir(parents=True)
        shutil.copy2(row["mesh"], folder / "figure.glb")
        if sha(folder / "figure.glb") != row["mesh_sha256"]:
            raise ValueError("Модель изменилась во время копирования")
        json_write(folder / "selection-provenance.json", row)
        if row["object"] in provenance_files:
            shutil.copy2(provenance_files[row["object"]], folder / "source-provenance.json")
        item = store.publish(folder, "figure", digest(row), metadata[row["object"]], row["object"])
        project.scenes.asset_versions[row["object"]] = item["id"]
        row["asset_id"] = item["id"]
    store.save(project)
    mapping = {
        "schema_version": 1,
        "name_source": "metadata/video_index.csv",
        "name_source_sha256": sha(repository / "metadata/video_index.csv"),
        "figures": records,
        "scale_basis": "nominal; default height 0.05 m, not a measurement",
    }
    json_write(destination / "class-map.json", mapping)
    json_write(destination / "gallery-selection.json", selection.model_dump())
    shutil.copy2(repository / "metadata/video_index.csv", destination / "video_index.csv")
    return {"root": str(destination), "class_count": len(records), "class_map": mapping}


def import_candidates(store, package, repository=None):
    repository = (repository or environment.REPOSITORY).resolve()
    package = package.resolve()
    project = store.load()
    figures = {f.id: f for f in project.figures}
    catalog = json.loads((package / "gallery/comparison-catalog.json").read_text())
    checked = []
    for reference in catalog:
        object_id = reference["object"]
        if object_id not in figures:
            continue
        path = (package / "gallery" / reference["file"]).resolve()
        if not path.is_relative_to(package):
            raise ValueError("Каталог выходит за пределы набора")
        data = json.loads(path.read_text())
        source_hash = sha(store.resolve(figures[object_id].video, exists=True))
        if data["source_video_sha256"] != source_hash:
            continue
        for entry in data["candidates"]:
            choice = Choice(
                object=object_id,
                id=entry["id"],
                mesh_sha256=entry["mesh_sha256"],
                source_video_sha256=source_hash,
            )
            rows, metadata, provenance = validate_selection(
                Selection(schema_version=1, choices=[choice]), package, repository
            )
            checked.append((rows[0], metadata[object_id], provenance.get(object_id)))
    if not checked:
        raise ValueError("В наборе нет кандидатов для исходных видео этого проекта")
    imported = []
    existing = store.artifacts("figure")
    for row, meta, provenance in checked:
        meta["quality_state"] = "imported_candidate"
        item = next(
            (
                a
                for a in existing
                if a["figure_id"] == row["object"]
                and a["manifest"]["metadata"].get("selection", {}).get("id") == row["candidate_id"]
                and a["manifest"]["metadata"].get("source_sha256") == row["source_video_sha256"]
                and a["manifest"]["files"].get(a["manifest"]["metadata"].get("mesh", "figure.glb"))
                == row["mesh_sha256"]
            ),
            None,
        )
        if item:
            store.artifact(item["id"])
        else:
            key = digest({"library": str(package), "candidate": row})
            folder = store.state / "import/library" / key
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row["mesh"], folder / "figure.glb")
            if sha(folder / "figure.glb") != row["mesh_sha256"]:
                raise ValueError("Модель изменилась во время копирования")
            json_write(folder / "selection-provenance.json", row)
            if provenance:
                shutil.copy2(provenance, folder / "source-provenance.json")
            meta.update(origin="candidate-library", quality_state="requires_visual_review")
            item = store.publish(folder, "figure", key, meta, row["object"])
            existing.append(item)
        imported.append(item["id"])
    return {"artifacts": imported, "count": len(imported)}
