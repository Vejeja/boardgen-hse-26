import copy
import json
from pathlib import Path

import pytest

from boardgen.maintenance import research_report as REPORT

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def published():
    return json.loads((ROOT / "research/yolo/results.json").read_text())


def test_published_experiment_and_readme_are_consistent(published):
    REPORT.check(published)


@pytest.mark.parametrize(
    "corruption",
    ["tp", "ap", "confidence", "manifest", "history", "background", "names", "search"],
)
def test_research_check_rejects_corrupted_evidence(published, corruption):
    data = copy.deepcopy(published)
    if corruption == "tp":
        data["test"]["operating_point"]["tp"] += 1
    elif corruption == "ap":
        data["test"]["metrics"]["metrics/mAP50(B)"] += 0.01
    elif corruption == "confidence":
        data["prediction"]["confidence"] = 0.4
    elif corruption == "manifest":
        data["dataset"]["manifest"]["files"]["data.yaml"] = "0" * 64
    elif corruption == "history":
        data["training"]["history"][-1]["epoch"] = "47"
    elif corruption == "background":
        sources = data["experiment"]["project"]["scenes"]["split_sources"]
        sources["test"]["boards"].append(sources["train"]["boards"][0])
    elif corruption == "search":
        data["validation"]["search"].pop()
    else:
        data["collection"]["name_source_sha256"] = "0" * 64
    with pytest.raises(AssertionError):
        REPORT.check(data)
