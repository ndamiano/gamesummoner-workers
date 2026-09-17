"""The box's instantiation-time values"""

import json
from pathlib import Path

ENV_FILE = Path("/etc/gamesummoner/env.json")


def load(path: Path = ENV_FILE) -> dict:
    return json.loads(Path(path).read_text())
