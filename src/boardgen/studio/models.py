from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Positive = Annotated[float, Field(gt=0, allow_inf_nan=False)]
Identifier = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$")]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class Clip(Model):
    start_s: float = Field(0, ge=0, allow_inf_nan=False)
    end_s: float | None = Field(None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def interval(self):
        if self.end_s is not None and self.end_s <= self.start_s:
            raise ValueError("Конец фрагмента должен быть позже начала")
        return self


class Figure(Model):
    id: Identifier
    name: str = Field(min_length=1, max_length=120)
    video: str = Field(min_length=1)
    clip: Clip = Field(default_factory=Clip)
    reviewed_views: dict[Literal["front", "left", "back", "right"], int] = Field(
        default_factory=dict
    )

    reviewed_frames_id: str | None = None
    front_view: int | None = Field(None, ge=0)
    period_views: int | None = Field(None, ge=5)
    focal_px: Positive | None = None


class Matte(Model):
    method: Literal["birefnet", "classic"] = "birefnet"
    input_size: int = Field(1024, ge=256, le=2048, multiple_of=32)
    cross_check: bool = False


class Shape(Model):
    method: Literal["mv", "spirula"] = "mv"
    extraction: Literal["native", "depth_fusion"] = "native"
    candidate_frames: int = Field(100, ge=10, le=400, multiple_of=5)
    frame_size: int = Field(1024, ge=256, le=2048)
    mesh_resolution: Literal[128, 256, 384] = 256
    timeout_seconds: int = Field(840, ge=60, le=86400)
    iterations: int = Field(5000, ge=1000, le=100000)
    cameras: Literal["orbit", "sfm"] = "orbit"
    profile: Literal["control", "detail"] = "control"
    cap: int = Field(120000, ge=10000, le=1000000)
    seed: int = Field(23, ge=0, le=2147483647)


class Appearance(Model):
    method: Literal["photo", "native", "neutral"] = "photo"
    photo_views: Literal[4, 12, 24] = 12
    registration: Literal["silhouette", "orbit"] = "silhouette"
    texture_size: Literal[1024, 2048, 4096] = 4096


class Recipe(Model):
    matte: Matte = Field(default_factory=Matte)
    shape: Shape = Field(default_factory=Shape)
    appearance: Appearance = Field(default_factory=Appearance)

    @model_validator(mode="after")
    def compatible(self):
        if self.shape.method == "mv" and self.shape.extraction != "native":
            raise ValueError("Depth fusion требует представление Spirula")
        if self.shape.extraction == "depth_fusion" and self.appearance.method == "native":
            raise ValueError("Depth fusion требует фототекстуру или нейтральный материал")
        if self.shape.method == "mv" and self.appearance.method == "native":
            raise ValueError("MV создаёт геометрию; выберите фототекстуру или нейтральный материал")
        if (
            self.shape.method == "spirula"
            and self.shape.cameras == "sfm"
            and self.appearance.registration == "orbit"
        ):
            raise ValueError("Регистрация orbit требует камеры orbit")
        return self


def merge(base: dict, patch: dict) -> dict:
    result = dict(base)
    for key, value in patch.items():
        result[key] = (
            merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


class Prepare(Model):
    defaults: Recipe = Field(default_factory=Recipe)
    overrides: dict[str, dict] = Field(default_factory=dict)
    reuse_completed: bool = True

    def resolved(self, figure_id: str) -> Recipe:
        return Recipe.model_validate(
            merge(self.defaults.model_dump(), self.overrides.get(figure_id, {}))
        )


class Board(Model):
    image: str
    width_m: Positive = 0.5
    height_m: Positive = 0.5


class Camera(Model):
    distance_m: tuple[Positive, Positive] = (0.8, 1.2)
    elevation_deg: tuple[float, float] = (25, 45)
    azimuth_deg: tuple[float, float] = (160, 340)
    fov_deg: float = Field(45, ge=35, le=65)
    jitter_xy_m: float = Field(0.03, ge=0, le=1)
    jitter_roll_deg: float = Field(4, ge=0, le=45)
    exposure_ev: float = Field(-1.4, ge=-10, le=10)

    @model_validator(mode="after")
    def ranges(self):
        for key in ("distance_m", "elevation_deg", "azimuth_deg"):
            lo, hi = getattr(self, key)
            if not all(math.isfinite(v) for v in (lo, hi)) or lo > hi:
                raise ValueError(f"{key}: диапазон должен возрастать")
        if not 1 <= self.elevation_deg[0] <= self.elevation_deg[1] <= 90:
            raise ValueError("Угол над полем: от 1 до 90°")
        if not -360 <= self.azimuth_deg[0] <= self.azimuth_deg[1] <= 360:
            raise ValueError("Азимут: от −360 до 360°")
        return self


class SceneSources(Model):
    boards: list[Board] = Field(default_factory=list)
    skyboxes: list[str] = Field(default_factory=list)


class Platform(Model):
    radius_m: Positive = 0.075
    height_m: Positive = 0.018
    elevation_deg: tuple[float, float] = (10, 50)
    frame_fraction: tuple[float, float] = (0.25, 0.60)

    @model_validator(mode="after")
    def ranges(self):
        if not 1 <= self.elevation_deg[0] <= self.elevation_deg[1] <= 90:
            raise ValueError("Подиум: угол камеры от 1 до 90°")
        if not 0.1 <= self.frame_fraction[0] <= self.frame_fraction[1] <= 0.8:
            raise ValueError("Подиум: доля кадра от 0.1 до 0.8")
        return self


class Scenes(Model):
    asset_versions: dict[str, Identifier] = Field(default_factory=dict)
    selected_classes: list[str] | None = None
    scale_basis: Literal["nominal", "shared_video"] = "nominal"
    reference_class: str | None = None
    count: int = Field(1000, ge=2, le=1000000)
    seed: int = Field(42, ge=0, le=2147483647)
    image_size: tuple[int, int] = (1280, 960)
    objects_per_scene: tuple[int, int] = (4, 12)
    negative_fraction: float = Field(0.05, ge=0, le=0.5)
    boards: list[Board] = Field(default_factory=list)
    skyboxes: list[str] = Field(default_factory=list)
    environment_strength: float = Field(1, ge=0, le=20)
    camera: Camera = Field(default_factory=Camera)
    nominal_height_m: Positive = 0.05
    height_overrides: dict[str, Positive] = Field(default_factory=dict)
    min_side_px: int = Field(12, ge=1, le=512)
    max_attempts: int = Field(10, ge=1, le=100)
    neutral_ablation: bool = False
    scene_type: Literal["board", "platform", "mixed"] = "board"
    platform: Platform = Field(default_factory=Platform)
    split_sources: dict[Literal["train", "val", "test"], SceneSources] = Field(default_factory=dict)
    render_samples: int = Field(64, ge=1, le=256)
    environment_rotation_deg: tuple[float, float] = (-180, 180)
    light_energy: tuple[Positive, Positive] = (180, 180)
    exposure_jitter_ev: float = Field(0, ge=0, le=3)
    balanced_classes: bool = False
    stratified_negatives: bool = False

    @model_validator(mode="after")
    def dimensions(self):
        if not all(64 <= v <= 8192 for v in self.image_size):
            raise ValueError("Размер изображения: от 64 до 8192 пикселей")
        if not 1 <= self.objects_per_scene[0] <= self.objects_per_scene[1] <= 100:
            raise ValueError("Число фигур: возрастающий диапазон от 1 до 100")
        for pair in (self.environment_rotation_deg, self.light_energy):
            if not all(math.isfinite(v) for v in pair) or pair[0] > pair[1]:
                raise ValueError("Диапазоны освещения должны возрастать")
        return self


class RealDataset(Model):
    path: str
    role: Literal["development", "test_only"] = "development"
    groups_csv: str | None = None
    negative_labels_confirmed: bool = False


class Dataset(Model):
    name: str = Field("tabletop", pattern=r"^[a-zA-Z0-9_-]{1,80}$")
    include_synthetic: bool = True
    real_sources: list[RealDataset] = Field(default_factory=list)
    val_fraction: float = Field(0.2, gt=0, lt=1)
    seed: int = Field(42, ge=0, le=2147483647)
    near_duplicates: Literal["review", "error"] = "review"
    test_fraction: float = Field(0, ge=0, lt=1)
    background_policy: Literal["none", "test_only", "all_splits"] = "none"

    @model_validator(mode="after")
    def fractions(self):
        if self.val_fraction + self.test_fraction >= 1:
            raise ValueError("Сумма долей val и test должна быть меньше 1")
        return self


class Training(Model):
    weights: str = "yolo11n.pt"
    epochs: int = Field(100, ge=1, le=10000)
    time: float | None = Field(None, gt=0, le=24, allow_inf_nan=False)
    imgsz: int = Field(640, ge=64, le=2048, multiple_of=32)
    batch: int = Field(4, ge=1, le=128)
    patience: int = Field(20, ge=0, le=1000)
    workers: int = Field(2, ge=0, le=16)
    optimizer: Literal["auto", "SGD", "Adam", "AdamW"] = "auto"
    lr0: float = Field(0.01, gt=0, le=1)
    seed: int = Field(42, ge=0, le=2147483647)
    deterministic: bool = True
    mosaic: float = Field(0.5, ge=0, le=1)
    cache: bool = False
    nbs: int = Field(64, ge=1, le=256)
    warmup_epochs: float = Field(3, ge=0, le=100)
    weight_decay: float = Field(0.0005, ge=0, le=1)
    cos_lr: bool = False
    close_mosaic: int = Field(10, ge=0, le=1000)
    hsv_h: float = Field(0.015, ge=0, le=1)
    hsv_s: float = Field(0.7, ge=0, le=1)
    hsv_v: float = Field(0.4, ge=0, le=1)
    fliplr: float = Field(0.5, ge=0, le=1)
    flipud: float = Field(0, ge=0, le=1)
    mixup: float = Field(0, ge=0, le=1)
    copy_paste: float = Field(0, ge=0, le=1)
    scale: float = Field(0.5, ge=0, le=1)
    translate: float = Field(0.1, ge=0, le=1)


class Evaluation(Model):
    split: Literal["val", "test"] = "val"
    confidence: float = Field(0.25, gt=0, lt=1)
    nms_iou: float = Field(0.7, gt=0, lt=1)
    matching_iou: float = Field(0.5, gt=0, lt=1)
    imgsz: int | None = Field(None, ge=64, le=2048, multiple_of=32)








class Project(Model):
    schema_version: Literal["boardgen.studio.v1"] = "boardgen.studio.v1"
    name: str = Field("Boardgen", min_length=1, max_length=120)
    figures: list[Figure] = Field(default_factory=list)
    prepare: Prepare = Field(default_factory=Prepare)
    scenes: Scenes = Field(default_factory=Scenes)
    dataset: Dataset = Field(default_factory=Dataset)
    training: Training = Field(default_factory=Training)
    evaluation: Evaluation = Field(default_factory=Evaluation)

    @model_validator(mode="after")
    def references(self):
        ids = [f.id for f in self.figures]
        names = [f.name.strip() for f in self.figures]
        if len(set(ids)) != len(ids) or len(set(names)) != len(names):
            raise ValueError("ID и названия фигур должны быть уникальны")
        if any(not n for n in names):
            raise ValueError("Название фигуры не может быть пустым")
        if (
            set(self.prepare.overrides) - set(ids)
            or set(self.scenes.height_overrides) - set(ids)
            or set(self.scenes.asset_versions) - set(ids)
        ):
            raise ValueError("Настройки ссылаются на отсутствующую фигуру")
        if self.scenes.selected_classes is not None and set(self.scenes.selected_classes) - set(
            ids
        ):
            raise ValueError("Выбран отсутствующий класс для сцен")
        if self.scenes.reference_class and self.scenes.reference_class not in ids:
            raise ValueError("Опорная фигура отсутствует")
        for figure in self.figures:
            recipe = self.prepare.resolved(figure.id)
            views = figure.reviewed_views
            if any(
                i < 0 or i >= recipe.shape.candidate_frames or i % 5 == 4 for i in views.values()
            ):
                raise ValueError(f"{figure.name}: ракурсы должны принадлежать fit-набору")
            if len(set(views.values())) != len(views):
                raise ValueError(f"{figure.name}: выберите разные ракурсы")
        return self
