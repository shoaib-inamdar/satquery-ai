# conftest.py — adds the backend app directory to sys.path for pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
