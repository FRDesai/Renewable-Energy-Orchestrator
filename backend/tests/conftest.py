import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchestrator.config_loader import load_config  # noqa: E402


@pytest.fixture(scope="session")
def config():
    return load_config()
