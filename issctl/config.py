import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "data" / "state.json"
SIM_STATE_FILE = ROOT / "data" / "state-sim.json"  # keep simulated calibration out of the real one


def _merge(base, override):
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            out[key] = _merge(base[key], value)
        else:
            out[key] = value
    return out


def load_config(path=None):
    """Your config layered over config.example.toml, so new keys get sensible defaults
    instead of raising KeyError on a config written before they existed."""
    with open(ROOT / "config.example.toml", "rb") as f:
        defaults = tomllib.load(f)
    path = Path(path) if path else ROOT / "config.toml"
    if not path.exists():
        return defaults
    with open(path, "rb") as f:
        return _merge(defaults, tomllib.load(f))


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
