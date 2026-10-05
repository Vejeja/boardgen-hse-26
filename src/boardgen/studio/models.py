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


























class Project(Model):
    schema_version: Literal["boardgen.studio.v1"] = "boardgen.studio.v1"
    name: str = Field("Boardgen", min_length=1, max_length=120)
    figures: list[Figure] = Field(default_factory=list)
    prepare: Prepare = Field(default_factory=Prepare)

    @model_validator(mode="after")
    def references(self):
        ids = [f.id for f in self.figures]
        names = [f.name.strip() for f in self.figures]
        if len(set(ids)) != len(ids) or len(set(names)) != len(names):
            raise ValueError("ID и названия фигур должны быть уникальны")
        if any(not n for n in names):
            raise ValueError("Название фигуры не может быть пустым")
        if (set(self.prepare.overrides) - set(ids)):
            raise ValueError("Настройки ссылаются на отсутствующую фигуру")
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
