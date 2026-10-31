import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RAW = ROOT / "data" / "raw"
needs_data = pytest.mark.skipif(not (RAW / "MANIFEST.csv").exists(), reason="data/raw is not present")
