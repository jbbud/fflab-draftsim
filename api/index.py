from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # repo root
sys.path.insert(0, str(ROOT / "src"))

from fflab.web import GuiHandler

class handler(GuiHandler):
    pass