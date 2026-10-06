import subprocess
import sys

import pytest
from pydantic import ValidationError

from boardgen.rendering.blender_render import BlenderSession
from boardgen.studio.models import Training
from boardgen.studio.parameters import Train


def test_time_limits_are_validated_and_serialized():
    assert Training(time=2.5).model_dump()["time"] == 2.5
    assert Train(deadline_unix=1000).model_dump()["deadline_unix"] == 1000
    for value in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValidationError):
            Training(time=value)
        with pytest.raises(ValidationError):
            Train(deadline_unix=value)


def test_session_reuses_then_recycles_process_and_propagates_failure(tmp_path, monkeypatch):
    original = subprocess.Popen
    program = (
        "import sys,json; "
        "[(print('BOARDGEN_RESULT '+json.dumps({'error':'bad scene'} if 'bad' in "
        "json.loads(line)['path'] else {'ok':True}),flush=True)) for line in sys.stdin]"
    )
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda args, **kwargs: original([sys.executable, "-u", "-c", program], **kwargs),
    )
    with BlenderSession("unused", recycle_after=2) as session:
        session.render(tmp_path / "one.json")
        first = session.process
        session.render(tmp_path / "two.json")
        assert session.process is first
        session.render(tmp_path / "three.json")
        assert first.poll() is not None
        assert session.process is not first
        with pytest.raises(RuntimeError, match="bad scene"):
            session.render(tmp_path / "bad.json")
        assert session.process is None
