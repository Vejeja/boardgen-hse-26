from __future__ import annotations

from pathlib import Path

from . import datasets, environment, rendering
from .models import Project
from .preparation import frames, prepare
from .runtime import work
from .storage import Artifact, Store, digest, sha


def selected_assets(store: Store, project: Project, parameters: dict) -> dict[str, Artifact]:
    explicit = project.scenes.asset_versions
    if any(explicit.get(k) != v for k, v in parameters.get("asset_ids", {}).items()):
        raise ValueError("Закрепите версии в проекте перед генерацией")
    from .workflow import selected_ids

    result = {}
    chosen = selected_ids(project)
    if not chosen:
        raise ValueError("Сначала закрепите версии в «Сравнение и выбор»")
    for figure in project.figures:
        if figure.id not in chosen:
            continue
        source_hash = sha(store.resolve(figure.video, exists=True))
        if figure.id not in explicit:
            raise ValueError(f"{figure.name}: закрепите версию в «Сравнение и выбор»")
        item = store.artifact(explicit[figure.id])
        if item:
            if (
                item["kind"] != "figure"
                or item["figure_id"] != figure.id
                or item["manifest"]["metadata"].get("source_sha256") != source_hash
            ):
                raise ValueError(f"{figure.name}: результат относится к другому исходному видео")
            result[figure.id] = item
    return result


def execute(store: Store, project: Project, job: dict, folder: Path) -> dict:
    parameters = job["parameters"]
    kind = job["kind"]
    if kind == "install":
        return environment.install(project)
    versions = environment.fingerprint()
    if kind == "prepare-candidates":
        from .candidates import prepare_candidates

        return prepare_candidates(store, project, job, folder, versions)
    if kind in {"frames", "prepare"}:
        ids = parameters.get("figure_ids", [f.id for f in project.figures])
        results = []
        for figure in project.figures:
            if figure.id in ids:
                stage = frames if kind == "frames" else prepare
                results.append(stage(store, project, figure, folder, versions)["id"])
        return {"artifacts": results}
    if kind in {"pilot", "dataset"}:
        if project.dataset.include_synthetic and project.scenes.selected_classes is not None:
            payload = project.model_dump(mode="json")
            chosen = set(project.scenes.selected_classes)
            if not chosen:
                raise ValueError("Выберите фигуры для сцен")
            payload["figures"] = [f for f in payload["figures"] if f["id"] in chosen]
            payload["prepare"]["overrides"] = {
                k: v for k, v in payload["prepare"]["overrides"].items() if k in chosen
            }
            payload["scenes"]["height_overrides"] = {
                k: v for k, v in payload["scenes"]["height_overrides"].items() if k in chosen
            }
            payload["scenes"]["asset_versions"] = {
                k: v for k, v in payload["scenes"]["asset_versions"].items() if k in chosen
            }
            if payload["scenes"]["reference_class"] not in chosen:
                payload["scenes"]["reference_class"] = None
            project = Project.model_validate(payload)
        temporary = work(folder, "output")
        records = []
        if project.dataset.include_synthetic:
            assets = selected_assets(store, project, parameters)
            count = min(4, project.scenes.count) if kind == "pilot" else project.scenes.count
            records = rendering.render(store, project, temporary / "render", assets, count)
        if kind == "pilot":
            if not records:
                raise ValueError("Пробный рендер требует подготовленные фигуры и игровые поля")
            meta = {
                "scenes": records,
                "project": project.model_dump(mode="json"),
                "versions": versions,
            }
        else:
            meta = datasets.build(store, project, temporary, records)
            meta = {k: v for k, v in meta.items() if k != "records"}
            meta.update({"project": project.model_dump(mode="json"), "versions": versions})
        return {
            "artifacts": [
                store.publish(
                    temporary,
                    kind,
                    digest(
                        {
                            "project": project.model_dump(mode="json"),
                            "parameters": parameters,
                            "versions": versions,
                        }
                    ),
                    meta,
                )["id"]
            ]
        }
    from .training import evaluate, predict, train

    fn = {"train": train, "evaluate": evaluate, "predict": predict}[kind]
    return fn(store, project, parameters, folder, versions)
