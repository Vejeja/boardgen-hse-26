from pydantic import Field, field_validator

from .candidates import VARIANTS, Variant
from .models import Identifier, Model


class Figures(Model):
    figure_ids: list[Identifier] | None = None


class Candidates(Figures):
    variants: list[Variant] = Field(
        default_factory=lambda: list(VARIANTS), min_length=1, max_length=3
    )

    @field_validator("variants")
    @classmethod
    def unique_ordered(cls, values):
        if len(values) != len(set(values)):
            raise ValueError("Методы не должны повторяться")
        return [v for v in VARIANTS if v in values]


class Render(Model):
    asset_ids: dict[str, Identifier] = Field(default_factory=dict)








PARAMETERS = {
    "frames": Figures,
    "prepare": Figures,
    "prepare-candidates": Candidates,
    "pilot": Render,
    "dataset": Render,
}


def validate(kind, values):
    if kind not in PARAMETERS:
        raise ValueError("Неизвестное действие")
    return PARAMETERS[kind].model_validate(values).model_dump(exclude_none=True)
