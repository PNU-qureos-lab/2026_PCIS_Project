"""Portable launcher that regenerates figures from saved Version 3 results."""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from render_results import main

if __name__ == "__main__":
    main()

