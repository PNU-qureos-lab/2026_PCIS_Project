"""Portable launcher for the Version 3 seven-stage workflow."""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from run_all import main

if __name__ == "__main__":
    main()

