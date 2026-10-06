from .models import Project
from .parameters import validate
from .storage import Store


def selected_ids(project):
    ids = project.scenes.selected_classes
    if ids is None:
        ids = list(project.scenes.asset_versions)
    return [f.id for f in project.figures if f.id in ids]


def dataset_inputs(project):
    chosen = selected_ids(project)
    scenes = project.scenes.model_dump(mode="json")
    scenes["selected_classes"] = chosen
    for key in ("asset_versions", "height_overrides"):
        scenes[key] = {k: v for k, v in scenes[key].items() if k in chosen}
    if scenes["reference_class"] not in chosen:
        scenes["reference_class"] = None
    return {
        "figures": [
            {"id": f.id, "name": f.name, "video": f.video}
            for f in project.figures
            if f.id in chosen
        ],
        "scenes": scenes,
        "dataset": project.dataset.model_dump(mode="json"),
    }


def preflight(store: Store, kind: str, parameters: dict, project=None):
    from .parameters import PARAMETERS

    if kind not in PARAMETERS:
        raise ValueError("Неизвестное действие")
    if kind == "train" and not parameters.get("resume_job_id") and not parameters.get("dataset_id"):
        raise ValueError("Выберите сохранённую версию датасета")
    if kind == "evaluate" and any(not parameters.get(k) for k in ("dataset_id", "model_id")):
        raise ValueError("Выберите модель и датасет с разметкой")
    if kind in {"predict"} and any(
        not parameters.get(k) for k in ("model_id", "source")
    ):
        raise ValueError("Выберите модель и источник изображений или видео")
    parameters = validate(kind, parameters)
    project = project or store.load()
    if kind == "train" and parameters.get("resume_job_id"):
        previous = store.job(parameters["resume_job_id"])
        checkpoint = store.state / "jobs" / previous["id"] / "training/resume-last.pt"
        if (
            previous["kind"] != "train"
            or previous["status"] not in {"failed", "interrupted", "cancelled"}
            or not checkpoint.is_file()
        ):
            raise ValueError("Для продолжения нужен остановленный запуск с сохранённой эпохой")
        if (
            parameters.get("dataset_id")
            and parameters["dataset_id"] != previous["parameters"]["dataset_id"]
        ):
            raise ValueError("Продолжение использует исходный датасет и настройки обучения")
        parameters["dataset_id"] = previous["parameters"]["dataset_id"]
        parameters["accept_near_duplicates"] = previous["parameters"].get(
            "accept_near_duplicates", False
        )
        project = Project.model_validate(previous["snapshot"])
    ids = parameters.get("figure_ids", [f.id for f in project.figures])
    if not isinstance(ids, list) or set(ids) - {f.id for f in project.figures}:
        raise ValueError("Выбрана отсутствующая фигура")
    if kind in {"frames", "prepare", "prepare-candidates"} and not ids:
        raise ValueError("Подключите видео фигур")
    if kind in {"frames", "prepare", "prepare-candidates"}:
        for f in project.figures:
            if f.id in ids:
                store.resolve(f.video, exists=True)
    if kind in {"prepare", "prepare-candidates"}:
        for f in project.figures:
            if f.id not in ids:
                continue
            recipes = [project.prepare.resolved(f.id)]
            if kind == "prepare-candidates":
                from .candidates import VARIANTS, candidate_project

                recipes = [
                    candidate_project(project, f.id, v).prepare.resolved(f.id)
                    for v in parameters.get("variants", VARIANTS)
                ]
            for recipe in recipes:
                if recipe.shape.method == "mv" and set(f.reviewed_views) != {
                    "front",
                    "left",
                    "back",
                    "right",
                }:
                    raise ValueError(f"{f.name}: выберите четыре ракурса после подготовки кадров")
                if (
                    recipe.shape.cameras == "orbit"
                    and (
                        recipe.shape.method == "spirula"
                        or recipe.appearance.registration == "orbit"
                    )
                    and (f.front_view is None or f.period_views is None or f.focal_px is None)
                ):
                    raise ValueError(
                        f"{f.name}: задайте передний ракурс, период оборота и фокус для orbit"
                    )
    if kind in {"pilot", "dataset"} and project.dataset.include_synthetic:
        from .stages import selected_assets

        if project.scenes.selected_classes is None:
            project = project.model_copy(deep=True)
            project.scenes.selected_classes = list(project.scenes.asset_versions)

        if project.scenes.selected_classes == [] or not project.figures:
            raise ValueError("Выберите модели в «Подготовка → Сравнение и выбор»")
        assets = selected_assets(store, project, parameters)
        if project.scenes.scale_basis == "shared_video":
            for figure in project.figures:
                if figure.id in assets and not assets[figure.id]["manifest"]["metadata"].get(
                    "frames_id"
                ):
                    raise ValueError(
                        f"{figure.name}: размеры по маскам требуют подготовленные кадры; задайте измеренную или базовую высоту"
                    )
        from .backgrounds import validate_sources

        validate_sources(store, project)
    if (
        kind == "dataset"
        and not project.dataset.include_synthetic
        and not project.dataset.real_sources
    ):
        raise ValueError("Подключите размеченные фотографии или включите Blender-сцены")
    if kind == "dataset":
        for source in project.dataset.real_sources:
            store.resolve(source.path, exists=True)
            if source.groups_csv:
                store.resolve(source.groups_csv, exists=True)
    if kind == "train" and not parameters.get("resume_job_id"):
        from .training import dataset

        data, _ = dataset(store, parameters["dataset_id"])
        if data["manifest"]["metadata"].get("near_duplicate_count") and not parameters.get(
            "accept_near_duplicates"
        ):
            raise ValueError("Просмотрите похожие пары датасета и подтвердите разделение")
    if kind in {"evaluate", "predict"}:
        model = store.artifact(parameters["model_id"], verify=False)
        if model["kind"] != "train":
            raise ValueError("Выберите опубликованную модель обучения")
        if kind == "evaluate":
            data = store.artifact(parameters["dataset_id"], verify=False)
            if data["kind"] != "dataset" or not data["manifest"]["metadata"].get("counts", {}).get(
                project.evaluation.split
            ):
                raise ValueError("В выбранном датасете отсутствует раздел оценки")
            source = store.artifact(model["manifest"]["metadata"]["dataset_id"], verify=False)
            if source["manifest"]["metadata"]["names"] != data["manifest"]["metadata"]["names"]:
                raise ValueError("Классы модели и датасета различаются")
    if kind in {"predict"}:
        store.resolve(parameters["source"], exists=True)
    return project, parameters
