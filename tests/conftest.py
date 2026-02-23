from pathlib import Path

import pytest
from hydra import initialize_config_dir, compose

CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "config")


@pytest.fixture()
def cfg():
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="default")
    return cfg
