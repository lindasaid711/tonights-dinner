import json
from pathlib import Path
from langgraph.checkpoint.memory import MemorySaver

DATA_DIR = Path("./data")
PROFILE_FILE = DATA_DIR / "user_profile.json"
EPISODIC_FILE = DATA_DIR / "episodic_memory.json"
METRICS_FILE = DATA_DIR / "metrics.json"


def load_json(path: Path, default=None):
    if path.exists():
        return json.loads(path.read_text())
    return default if default is not None else {}


def save_json(path: Path, data):
    DATA_DIR.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))


checkpointer = MemorySaver()
