# Laplace FEM assignment

The primary submission is `solver.c`, the three `mesh_*.msh` files and
`report.pdf`. The editable report is `report.md`. `results/` retains actual
measurements, SOR-search trials, CSR matrices, right-hand sides and solutions.

## Build and run

No service, credentials or GPU are required. The C solver needs GCC (or another
C11 compiler), libc, libm and libdl on Linux. It directly reads Gmsh 2.2 ASCII
triangular meshes. All FEM assembly and all five iterative methods are in C.

```sh
gcc -O3 -march=native -std=c11 -Wall -Wextra solver.c -lm -ldl -o solver
mkdir -p results
./solver mesh_100.msh results
./solver mesh_1000.msh results
./solver mesh_10000.msh results
```

These commands use portable dense C loops. The submitted measurements use
single-thread OpenBLAS for dense matrix-vector products; CSR and SOR use C
loops. `./run.sh` selects the installed NumPy ILP64 OpenBLAS library and runs
all benchmarks. It requires NumPy; `make_report.py` additionally requires
SciPy, Matplotlib and ReportLab. No compiler fast-math option is used.
For another supported ILP64 OpenBLAS installation set `DENSE_BLAS` to its
shared library path. Do not set it for an incompatible LP64 BLAS. Leave it
unset for the portable implementation. Times depend on machine and backend.

## Regenerate meshes and report

The three `.msh` files are already supplied; Gmsh is unnecessary to solve them.
`generate_meshes.py` uses Gmsh 4.15.2 and deterministic NumPy seeds to place
exactly N points and asks Gmsh to triangulate the square without inserting
additional points. The boundary is divided into equal straight segments;
interior points are stratified and jittered, with distributed omissions to
reach the exact count. This is an unstructured triangular mesh, not a Q4 mesh.
Install dependencies in a virtual environment if needed:

```sh
python3 -m venv /tmp/laplace-venv
/tmp/laplace-venv/bin/pip install -r requirements.txt
/tmp/laplace-venv/bin/python generate_meshes.py
/tmp/laplace-venv/bin/python validate.py
/tmp/laplace-venv/bin/python make_report.py
```

Gmsh's Linux wheel needs libGLU.so.1. In the prepared cloud environment Gmsh
is in `/workspace/setup-python`, already referenced by `generate_meshes.py`.
The independent `validate.py` reassembles every element, checks area and the
symmetrization, solves with SciPy's sparse direct solver, and estimates the
smallest eigenvalue. Run it after the C benchmarks complete, outside timed
runs. Report generation reads recorded results; it does not invent timings.

## Numerical conventions

* PDE: -div(grad T)=0 on the unit square. P1 triangles, exact element stiffness.
* Bottom/left are 0; top/right are 1. The two contradictory corner values at
  (1,0) and (0,1) are 0.5. This convention is explicitly recorded in the report.
* Original A: retain all interior rows of the stiffness matrix, replace each
  boundary row by its unit row, and set b_B=g, b_I=0. A is nonsymmetric.
* Original initial guess: x_B=g, x_I=0. Boundary residuals stay zero so SD/CG
  operate on the SPD interior block. This avoids claiming CG works on every
  nonsymmetric matrix. Symmetrized initial guess: zero.
* Symmetrization is applied to this row-modified full A: As=(A+A^T)/2.
  The same saved b is used. It changes the equations and boundary temperatures.
* Common stopping criterion: true ||b-Ax||_2/||b||_2 <= 1e-8, at most 100000
  iterations. SD, MR, CG and BiCGSTAB update recursive residuals, then verify
  the true residual before accepting convergence. SOR recomputes each sweep.
* MR means the one-direction minimal-residual iteration:
  alpha=(Ar,r)/(Ar,Ar), x+=alpha*r; it is not GMRES or MINRES.
* SOR: coarse omega=1.00,1.05,...,1.95, then local 0.005 search around the
  best coarse value (clipped to [1,1.995]); minimize iterations, choose lower
  omega on ties. "Optimum" means best on this search, not a proven continuum
  optimum. Original and symmetrized systems are tuned independently in CSR.
  Trial runs are capped at best_iterations+1 after the first convergence;
  "maxiter" trials then cannot beat the current best. Search time is excluded.
* Timings include initial/final residual products and the iteration loop,
  excluding parsing, assembly, allocation, CSR conversion, tuning and output.
  Five repetitions for N=100, three for N=1000 (median), one for N=10000.
  Every repetition starts from the prescribed initial guess.
* CSR uses sorted zero-based column indices, 32-bit integers, 64-bit values.
  `.csr`: first line N nnz, then row pointers, columns and values on separate
  lines. `.mtx` is an additional Matrix Market exchange format.

The supplied reference PDF guides the six-section report layout only; its
meshes, numbers, authorship and results are not reused.

The 10,000-node original-system CSV retains its completed log measurements
at six decimal places for time and three significant digits for residual.
Convergence was checked with the unrounded residual. The As preparation
avoids writing unchanged coefficients, without changing any matrix values.
`--sym-only` as a fourth solver argument resumes only symmetrized runs and
appends them to existing CSVs; use it only when original runs are complete.

Reported memory sizes are representation payloads, not peak process memory;
the benchmarking program keeps a dense reference while exercising CSR.
