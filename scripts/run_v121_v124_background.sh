#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/nakato/miniconda3/envs/Sign-env/bin/python"
RUN_ROOT="${ROOT}/outputs/background/v121_v124"
STATUS="${RUN_ROOT}/status.txt"

mkdir -p "${RUN_ROOT}"
cd "${ROOT}"

NVIDIA_LIBS="$("${PYTHON}" -c '
import site
from pathlib import Path
root = Path(site.getsitepackages()[0]) / "nvidia"
print(":".join(str(p / "lib") for p in root.iterdir() if (p / "lib").is_dir()))
')"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

stage() {
    printf '%s | %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$1" | tee "${STATUS}"
}

failed() {
    code=$?
    printf '%s | FAILED exit=%s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${code}" | tee "${STATUS}"
    exit "${code}"
}
trap failed ERR

stage "v121 replicas seed 42/101"
"${PYTHON}" scripts/run_isolated_experiments.py v121

stage "v122 extraccion HDF5 240 frames"
"${PYTHON}" scripts/build_dataset1_v122_h5.py

stage "v122-v124 pipeline escalonado"
"${PYTHON}" scripts/run_isolated_experiments.py staged

stage "LOSO 10 signers sobre v123"
"${PYTHON}" scripts/run_isolated_experiments.py loso \
    --config config/experiment/cls_v123.toml

stage "COMPLETE"
