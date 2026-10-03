from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .storage import sha

PACKAGE = Path(__file__).resolve().parents[1]
CHECKOUT = PACKAGE.parents[1] if (PACKAGE.parents[1] / "pyproject.toml").exists() else None
REPOSITORY = (
    Path(os.environ.get("BOARDGEN_HOME", str(CHECKOUT or Path.home() / ".cache/boardgen")))
    .expanduser()
    .resolve()
)












