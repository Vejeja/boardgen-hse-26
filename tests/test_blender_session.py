import subprocess
import sys

import pytest

from boardgen.rendering.blender_render import BlenderSession




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
