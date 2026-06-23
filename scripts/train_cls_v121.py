"""Atajo compatible para lanzar v121."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    command = [
        sys.executable,
        str(Path(__file__).with_name("train_isolated_staged.py")),
        "--config",
        "config/experiment/cls_v121.toml",
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(command))
