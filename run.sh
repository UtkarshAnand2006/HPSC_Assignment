#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
# Optional: use the verified, installed NumPy OpenBLAS in single-thread mode.
export DENSE_BLAS="$(python3 - <<'PY'
import numpy,glob
print(glob.glob(str(numpy.__path__[0])+'/../numpy.libs/*openblas*')[0])
PY
)"
export OPENBLAS_NUM_THREADS=1
gcc -O3 -march=native -std=c11 -Wall -Wextra solver.c -lm -ldl -o solver
for n in 100 1000 10000; do
    ./solver "mesh_${n}.msh" results > "results/log_${n}.txt" 2>&1
done
python3 validate.py
python3 make_report.py
