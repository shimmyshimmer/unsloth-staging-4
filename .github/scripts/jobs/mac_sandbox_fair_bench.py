"""Apple Silicon staging legs only run mac_* / mlx_* jobs: this runs sandbox_fair_bench.py there."""

import runpy
import sys
from pathlib import Path

target = Path(__file__).with_name("sandbox_fair_bench.py")
sys.argv[0] = str(target)
runpy.run_path(str(target), run_name = "__main__")
