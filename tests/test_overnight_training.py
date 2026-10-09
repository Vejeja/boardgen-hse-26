import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from boardgen.maintenance import overnight_training as runner


def test_prediction_list_inputs_are_explicitly_bounded_and_fully_consumed():
    calls = []

    class Values:
        def __init__(self, rows):
            self.rows = rows

        def cpu(self):
            return self

        def tolist(self):
            return self.rows

    class Model:
        def predict(self, paths, **kwargs):
            calls.append((paths, kwargs))

            assert len(paths) <= 4
            for path in paths:
                yield SimpleNamespace(
                    path=path,
                    boxes=SimpleNamespace(
                        xywhn=Values([[0.5, 0.5, 0.2, 0.2]]),
                        cls=Values([2]),
                        conf=Values([0.9]),
                    ),
                )

    paths = [f"/images/image_{i}.png" for i in range(11)]
    predictions = runner.predict_batches(Model(), paths, size=960, device="mps")
    assert [len(paths) for paths, _ in calls] == [4, 4, 3]
    assert [row["image"] for row in predictions] == [Path(p).name for p in paths]
    assert all(kwargs["imgsz"] == 960 and kwargs["device"] == "mps" for _, kwargs in calls)
    assert predictions[-1]["boxes"] == [[0.5, 0.5, 0.2, 0.2]]
    assert predictions[-1]["classes"] == [2]
    assert predictions[-1]["confidence"] == [0.9]


def test_setup_requires_source_before_creating_output(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(runner.environment, "REPOSITORY", tmp_path)
    monkeypatch.setattr(sys, "argv", ["runner", "setup", "--project", str(tmp_path / "project")])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
    assert "setup requires --source" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_setup_forwards_explicit_source(tmp_path, monkeypatch):
    calls = []
    source = tmp_path / "source"
    project = tmp_path / "project"
    monkeypatch.setattr(runner.environment, "REPOSITORY", tmp_path)
    monkeypatch.setattr(runner, "setup", lambda a, b: calls.append((a, b)))
    monkeypatch.setattr(
        sys, "argv", ["runner", "setup", "--project", str(project), "--source", str(source)]
    )
    runner.main()
    assert calls == [(source, project)]


def test_finish_does_not_require_source(tmp_path, monkeypatch):
    calls = []
    project = tmp_path / "project"
    monkeypatch.setattr(runner.environment, "REPOSITORY", tmp_path)
    monkeypatch.setattr(runner, "Store", lambda path: path)
    monkeypatch.setattr(
        runner, "resume_evaluation", lambda store, output: calls.append((store, output))
    )
    monkeypatch.setattr(sys, "argv", ["runner", "finish", "--project", str(project)])
    runner.main()
    assert calls == [(project, tmp_path / "out" / "project")]
