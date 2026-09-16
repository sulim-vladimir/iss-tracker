import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "data" / "state.json"
SIM_STATE_FILE = ROOT / "data" / "state-sim.json"  # keep simulated calibration out of the real one


def load_config(path=None):
    path = Path(path) if path else ROOT / "config.toml"
    if not path.exists():
        path = ROOT / "config.example.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_state(path=None):
    """Persistent calibration results (sync offsets, camera Jacobians, boresights)."""
    path = path or STATE_FILE
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_state(state, path=None):
    path = path or STATE_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))
