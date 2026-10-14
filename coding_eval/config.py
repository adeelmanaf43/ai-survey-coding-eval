"""Project settings, read once from config.toml in the project folder."""
from pathlib import Path

try:
    import tomllib                    # built into Python 3.11 and newer
except ModuleNotFoundError:
    import tomli as tomllib           # the same reader for Python 3.10 and older (in requirements.txt)

ROOT = Path(__file__).resolve().parent.parent          # the project folder
with open(ROOT / "config.toml", "rb") as f:
    CONFIG = tomllib.load(f)

SEED = CONFIG["seed"]
SCHEMES = CONFIG["schemes"]
PROCESSED = ROOT / CONFIG["paths"]["processed"]
RESULTS = ROOT / CONFIG["paths"]["results"]
CODEBOOKS = ROOT / CONFIG["paths"]["codebooks"]
