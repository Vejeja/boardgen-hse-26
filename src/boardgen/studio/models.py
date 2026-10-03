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







































class Project(Model):
    schema_version: Literal["boardgen.studio.v1"] = "boardgen.studio.v1"
    name: str = Field("Boardgen", min_length=1, max_length=120)
    figures: list[Figure] = Field(default_factory=list)

    @model_validator(mode="after")
    def references(self):
        ids = [f.id for f in self.figures]
        names = [f.name.strip() for f in self.figures]
        if len(set(ids)) != len(ids) or len(set(names)) != len(names):
            raise ValueError("ID и названия фигур должны быть уникальны")
        if any(not n for n in names):
            raise ValueError("Название фигуры не может быть пустым")
        return self
