/* P1 FEM Laplace, Gmsh 2.2 ASCII, dense/CSR solvers. See README.md.
   gcc -O3 -march=native -std=c11 solver.c -lm -ldl -o solver
   ./solver mesh_100.msh results
   Optional DENSE_BLAS=/path/to/libscipy_openblas64_*.so uses one-thread
   OpenBLAS dgemv; without it, portable dense loops are used. */
#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define TOL 1e-8
#define MAXIT 100000
static void *alloc(size_t n, size_t s) {
  void *p = calloc(n, s);
  if (!p) {
    fprintf(stderr, "Allocation failed\n");
    exit(1);
  }
  return p;
}
static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}
typedef struct {
  int n, nt, nb;
  double *xy, *g;
  int *tri, *boundary;
} Mesh;
typedef struct {
  int n, nnz, *rp, *ci;
  double *v, *d, *diag;
  int csr;
} Matrix;
typedef struct {
  int it;
  double sec, res;
  const char *status;
} Result;
/* Optional NumPy ILP64 OpenBLAS ABI, dynamically loaded to keep the code
 * portable. */
typedef void (*Gemv)(int, int, int64_t, int64_t, double, const double *,
                     int64_t, const double *, int64_t, double, double *,
                     int64_t);
static Gemv blas = NULL;
static void init_blas(void) {
  const char *p = getenv("DENSE_BLAS");
  if (!p)
    return;
  void *h = dlopen(p, RTLD_NOW | RTLD_LOCAL);
  if (!h) {
    fprintf(stderr, "%s\n", dlerror());
    exit(1);
  }
  blas = (Gemv)dlsym(h, "scipy_cblas_dgemv64_");
  void (*threads)(int) =
      (void (*)(int))dlsym(h, "scipy_openblas_set_num_threads64_");
  if (!blas || !threads) {
    fprintf(stderr, "Unsupported BLAS ABI\n");
    exit(1);
  }
  threads(1);
}
static Mesh read_mesh(const char *path) {
  FILE *f = fopen(path, "r");
  if (!f) {
    perror(path);
    exit(1);
  }
  Mesh m = {0};
  char *line = NULL;
  size_t cap = 0;
  int *map = NULL, maxid = 0;
  while (getline(&line, &cap, f) > 0) {
    if (!strncmp(line, "$Nodes", 6)) {
      getline(&line, &cap, f);
      m.n = atoi(line);
      m.xy = alloc(2 * m.n, sizeof(double));
      m.g = alloc(m.n, sizeof(double));
      m.boundary = alloc(m.n, sizeof(int));
      int *tags = alloc(m.n, sizeof(int));
      for (int i = 0; i < m.n; i++) {
        double z;
        getline(&line, &cap, f);
        if (sscanf(line, "%d%lf%lf%lf", &tags[i], &m.xy[2 * i],
                   &m.xy[2 * i + 1], &z) != 4)
          exit(1);
        if (tags[i] > maxid)
          maxid = tags[i];
      }
      map = alloc(maxid + 1, sizeof(int));
      for (int i = 0; i <= maxid; i++)
        map[i] = -1;
      for (int i = 0; i < m.n; i++)
        map[tags[i]] = i;
      free(tags);
    } else if (!strncmp(line, "$Elements", 9)) {
      getline(&line, &cap, f);
      int ne = atoi(line);
      m.tri = alloc(3 * ne, sizeof(int));
      for (int i = 0; i < ne; i++) {
        getline(&line, &cap, f);
        char *p = line;
        strtol(p, &p, 10);
        int type = strtol(p, &p, 10), ntags = strtol(p, &p, 10);
        for (int k = 0; k < ntags; k++)
          strtol(p, &p, 10);
        if (type == 2) {
          for (int k = 0; k < 3; k++) {
            int tag = strtol(p, &p, 10);
            if (tag < 1 || tag > maxid || map[tag] < 0)
              exit(1);
            m.tri[3 * m.nt + k] = map[tag];
          }
          m.nt++;
        }
      }
    }
  }
  fclose(f);
  free(line);
  free(map);
  if (!m.n || !m.nt) {
    fprintf(stderr, "No nodes/triangles\n");
    exit(1);
  }
  for (int i = 0; i < m.n; i++) {
    double x = m.xy[2 * i], y = m.xy[2 * i + 1];
    if (x < 1e-12 || y < 1e-12 || x > 1 - 1e-12 || y > 1 - 1e-12) {
      m.boundary[i] = 1;
      m.nb++; /* conflicting corners take arithmetic mean */
      int zero = (x < 1e-12 || y < 1e-12),
          one = (x > 1 - 1e-12 || y > 1 - 1e-12);
      m.g[i] = (zero && one) ? .5 : one;
    }
  }
  return m;
}
static Matrix assemble(Mesh *m, double **bout) {
  int n = m->n;
  Matrix a = {0};
  a.n = n;
  a.d = alloc((size_t)n * n, sizeof(double));
  double *b = alloc(n, sizeof(double));
  for (int t = 0; t < m->nt; t++) {
    int *id = m->tri + 3 * t;
    double x[3], y[3];
    for (int k = 0; k < 3; k++) {
      x[k] = m->xy[2 * id[k]];
      y[k] = m->xy[2 * id[k] + 1];
    }
    double twice =
        (x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]);
    if (fabs(twice) < 1e-15) {
      fprintf(stderr, "Degenerate element\n");
      exit(1);
    }
    double bx[3] = {y[1] - y[2], y[2] - y[0], y[0] - y[1]},
           by[3] = {x[2] - x[1], x[0] - x[2], x[1] - x[0]};
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        a.d[(size_t)id[i] * n + id[j]] +=
            (bx[i] * bx[j] + by[i] * by[j]) / (2 * fabs(twice));
  }
  for (int i = 0; i < n; i++)
    if (m->boundary[i]) {
      memset(a.d + (size_t)i * n, 0, n * sizeof(double));
      a.d[(size_t)i * n + i] = 1;
      b[i] = m->g[i];
    }
  *bout = b;
  return a;
}
static void to_csr(Matrix *a) {
  int n = a->n;
  free(a->rp);
  free(a->ci);
  free(a->v);
  free(a->diag);
  a->rp = alloc(n + 1, sizeof(int));
  a->diag = alloc(n, sizeof(double));
  int count = 0;
  for (int i = 0; i < n; i++) {
    a->rp[i] = count;
    for (int j = 0; j < n; j++)
      if (a->d[(size_t)i * n + j] != 0)
        count++;
    a->diag[i] = a->d[(size_t)i * n + i];
  }
  a->rp[n] = count;
  a->nnz = count;
  a->ci = alloc(count, sizeof(int));
  a->v = alloc(count, sizeof(double));
  count = 0;
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++) {
      double v = a->d[(size_t)i * n + j];
      if (v != 0) {
        a->ci[count] = j;
        a->v[count++] = v;
      }
    }
}
static void mv(const Matrix *a, const double *x, double *y) {
  int n = a->n;
  if (a->csr) {
    for (int i = 0; i < n; i++) {
      double s = 0;
      for (int k = a->rp[i]; k < a->rp[i + 1]; k++)
        s += a->v[k] * x[a->ci[k]];
      y[i] = s;
    }
  } else if (blas) {
    blas(101, 111, n, n, 1, a->d, n, x, 1, 0, y, 1);
  } else {
    for (int i = 0; i < n; i++) {
      double s = 0;
      const double *row = a->d + (size_t)i * n;
      for (int j = 0; j < n; j++)
        s += row[j] * x[j];
      y[i] = s;
    }
  }
}
static double dot(int n, const double *x, const double *y) {
  double s = 0;
  for (int i = 0; i < n; i++)
    s += x[i] * y[i];
  return s;
}
static double residual(Matrix *a, const double *b, const double *x, double *r,
                       double bn) {
  mv(a, x, r);
  for (int i = 0; i < a->n; i++)
    r[i] = b[i] - r[i];
  return sqrt(dot(a->n, r, r)) / bn;
}
/* method: 0 SOR, 1 SD, 2 one-step minimum residual (MR), 3 CG, 4 BiCGSTAB.
   Original starts at x_B=g, x_I=0, an invariant SPD interior subspace.
   Symmetrized starts at zero. No preconditioner is applied. */
static Result solve(Matrix *a, double *b, Mesh *m, int sym, int method,
                    double omega, int cap, double *out) {
  int n = a->n;
  if (n <= 0) {
    fprintf(stderr, "Invalid matrix size\n");
    exit(1);
  }
  double *w = alloc((size_t)10 * n, sizeof(double));
  double *x = w, *r = x + n, *p = r + n, *v = p + n, *s = v + n, *t = s + n,
         *rh = t + n, *tmp = rh + n;
  double bn = sqrt(dot(n, b, b)), alpha = 1, beta = 0, rho = 1, oldrho = 1,
         om = 1, rr = 0;
  if (!sym)
    for (int i = 0; i < n; i++)
      if (m->boundary[i])
        x[i] = m->g[i];
  double begin = now();
  double rel = residual(a, b, x, r, bn);
  memcpy(p, r, n * sizeof(double));
  memcpy(rh, r, n * sizeof(double));
  rr = dot(n, r, r);
  int it = 0;
  const char *status = "maxiter";
  for (it = 1; it <= cap; it++) {
    if (method == 0) {
      for (int i = 0; i < n; i++) {
        double z = 0;
        if (a->csr) {
          for (int k = a->rp[i]; k < a->rp[i + 1]; k++)
            if (a->ci[k] != i)
              z += a->v[k] * x[a->ci[k]];
        } else {
          const double *row = a->d + (size_t)i * n;
          for (int j = 0; j < i; j++)
            z += row[j] * x[j];
          for (int j = i + 1; j < n; j++)
            z += row[j] * x[j];
        }
        x[i] = (1 - omega) * x[i] + omega * (b[i] - z) / a->diag[i];
      }
      rel = residual(a, b, x, r, bn);
    } else if (method == 1 || method == 2) {
      mv(a, r, v);
      double rv = dot(n, r, v), den = method == 1 ? rv : dot(n, v, v);
      if (fabs(den) < 1e-300) {
        status = "breakdown";
        break;
      }
      alpha = (method == 1 ? dot(n, r, r) : rv) / den;
      for (int i = 0; i < n; i++) {
        x[i] += alpha * r[i];
        r[i] -= alpha * v[i];
      }
      rel = sqrt(dot(n, r, r)) / bn;
    } else if (method == 3) {
      mv(a, p, v);
      double den = dot(n, p, v);
      if (fabs(den) < 1e-300) {
        status = "breakdown";
        break;
      }
      alpha = rr / den;
      for (int i = 0; i < n; i++) {
        x[i] += alpha * p[i];
        r[i] -= alpha * v[i];
      }
      double rrnew = dot(n, r, r);
      beta = rrnew / rr;
      for (int i = 0; i < n; i++)
        p[i] = r[i] + beta * p[i];
      rr = rrnew;
      rel = sqrt(rr) / bn;
    } else {
      rho = dot(n, rh, r);
      if (fabs(rho) < 1e-300 || fabs(om) < 1e-300) {
        status = "breakdown";
        break;
      }
      if (it == 1)
        memcpy(p, r, n * sizeof(double));
      else {
        beta = (rho / oldrho) * (alpha / om);
        for (int i = 0; i < n; i++)
          p[i] = r[i] + beta * (p[i] - om * v[i]);
      }
      mv(a, p, v);
      double den = dot(n, rh, v);
      if (fabs(den) < 1e-300) {
        status = "breakdown";
        break;
      }
      alpha = rho / den;
      for (int i = 0; i < n; i++)
        s[i] = r[i] - alpha * v[i];
      if (sqrt(dot(n, s, s)) / bn <= TOL) {
        for (int i = 0; i < n; i++)
          x[i] += alpha * p[i];
        rel = residual(a, b, x, r, bn);
        if (rel <= TOL) {
          status = "converged";
          break;
        } /* residual replacement */
        oldrho = rho;
        continue;
      }
      mv(a, s, t);
      den = dot(n, t, t);
      if (den < 1e-300) {
        status = "breakdown";
        break;
      }
      om = dot(n, t, s) / den;
      for (int i = 0; i < n; i++) {
        x[i] += alpha * p[i] + om * s[i];
        r[i] = s[i] - om * t[i];
      }
      oldrho = rho;
      rel = sqrt(dot(n, r, r)) / bn;
    }
    if (!isfinite(rel) || rel > 1e12) {
      status = "diverged";
      break;
    }
    if (rel <= TOL) {
      rel = residual(a, b, x, tmp, bn);
      if (rel <= TOL) {
        status = "converged";
        break;
      }
      if (method != 0) {
        memcpy(r, tmp, n * sizeof(double));
        if (method == 3) {
          memcpy(p, r, n * sizeof(double));
          rr = dot(n, r, r);
        }
      }
    }
  }
  if (it > cap) {
    it = cap;
  }
  rel = residual(a, b, x, tmp, bn);
  double sec = now() - begin;
  if (out)
    memcpy(out, x, n * sizeof(double));
  free(w);
  return (Result){it, sec, rel, status};
}
static double tune(Matrix *a, double *b, Mesh *m, int sym, FILE *f) {
  a->csr = 1;
  int best = MAXIT;
  double bw = 1;
  for (int stage = 0; stage < 2; stage++) {
    double lo = stage ? fmax(1, bw - .05) : 1,
           hi = stage ? fmin(1.995, bw + .05) : 1.95, step = stage ? .005 : .05;
    for (double om = lo; om <= hi + 1e-10; om += step) {
      Result r =
          solve(a, b, m, sym, 0, om, best == MAXIT ? 20000 : best + 1, NULL);
      fprintf(f, "%d,%s,%.6f,%d,%s\n", m->n, sym ? "symmetrized" : "original",
              om, r.it, r.status);
      if (!strcmp(r.status, "converged") &&
          (r.it < best || (r.it == best && om < bw))) {
        best = r.it;
        bw = om;
      }
    }
  }
  fflush(f);
  return bw;
}
static void save_matrix(Matrix *a, double *b, const char *dir, int sym) {
  char path[512];
  snprintf(path, sizeof(path), "%s/A%s_%d.mtx", dir, sym ? "s" : "", a->n);
  FILE *f = fopen(path, "w");
  if (!f) {
    perror(path);
    exit(1);
  }
  fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n%d %d %d\n",
          a->n, a->n, a->nnz);
  for (int i = 0; i < a->n; i++)
    for (int k = a->rp[i]; k < a->rp[i + 1]; k++)
      fprintf(f, "%d %d %.17g\n", i + 1, a->ci[k] + 1, a->v[k]);
  fclose(f);
  if (!sym) {
    snprintf(path, sizeof(path), "%s/b_%d.txt", dir, a->n);
    f = fopen(path, "w");
    for (int i = 0; i < a->n; i++)
      fprintf(f, "%.17g\n", b[i]);
    fclose(f);
  }
  /* Actual CSR serialization: n nnz; row pointers; columns; values, all
   * zero-based. */
  snprintf(path, sizeof(path), "%s/A%s_%d.csr", dir, sym ? "s" : "", a->n);
  f = fopen(path, "w");
  fprintf(f, "%d %d\n", a->n, a->nnz);
  for (int i = 0; i <= a->n; i++)
    fprintf(f, "%d%c", a->rp[i], i == a->n ? '\n' : ' ');
  for (int k = 0; k < a->nnz; k++)
    fprintf(f, "%d%c", a->ci[k], k == a->nnz - 1 ? '\n' : ' ');
  for (int k = 0; k < a->nnz; k++)
    fprintf(f, "%.17g%c", a->v[k], k == a->nnz - 1 ? '\n' : ' ');
  fclose(f);
}
static int cmpd(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}
int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s mesh.msh output-directory\n", argv[0]);
    return 1;
  }
  init_blas();
  Mesh m = read_mesh(argv[1]);
  double *b;
  Matrix a = assemble(&m, &b);
  char path[512];
  snprintf(path, sizeof(path), "%s/runs_%d.csv", argv[2], m.n);
  int sym_only = argc > 3 && !strcmp(argv[3], "--sym-only");
  FILE *f = fopen(path, sym_only ? "a" : "w");
  if (!f) {
    perror(path);
    return 1;
  }
  if (!sym_only)
    fprintf(f,
            "N,elements,boundary,system,storage,method,omega,iterations,time_"
            "s,relative_residual,status,nnz,repeats\n");
  snprintf(path, sizeof(path), "%s/tuning_%d.csv", argv[2], m.n);
  FILE *tf = fopen(path, sym_only ? "a" : "w");
  if (!sym_only)
    fprintf(tf, "N,system,omega,iterations,status\n");
  const char *names[] = {"SOR", "SD", "MR", "CG", "BiCGSTAB"};
  double *x = alloc(m.n, sizeof(double));
  int failed = 0;
  for (int sym = sym_only ? 1 : 0; sym < 2; sym++) {
    if (sym)
      for (int i = 0; i < a.n; i++)
        for (int j = i + 1; j < a.n; j++) {
          /* Avoid touching identical entries (especially calloc zero pages). */
          if (a.d[(size_t)i * a.n + j] == a.d[(size_t)j * a.n + i])
            continue;
          double z = .5 * (a.d[(size_t)i * a.n + j] + a.d[(size_t)j * a.n + i]);
          a.d[(size_t)i * a.n + j] = a.d[(size_t)j * a.n + i] = z;
        }
    to_csr(&a);
    save_matrix(&a, b, argv[2], sym);
    double omega = tune(&a, b, &m, sym, tf);
    fprintf(stderr, "N=%d %s nnz=%d tuned omega=%.3f\n", m.n,
            sym ? "symmetrized" : "original", a.nnz, omega);
    for (int storage = 0; storage < 2; storage++) {
      a.csr = storage;
      for (int method = 0; method < 5; method++) {
        int reps = m.n < 1000 ? 5 : (m.n < 10000 ? 3 : 1);
        double times[5];
        Result r = {0};
        for (int rep = 0; rep < reps; rep++) {
          r = solve(&a, b, &m, sym, method, omega, MAXIT, x);
          times[rep] = r.sec;
        }
        qsort(times, reps, sizeof(double), cmpd);
        r.sec = times[reps / 2];
        if (strcmp(r.status, "converged"))
          failed++;
        fprintf(f, "%d,%d,%d,%s,%s,%s,%.6f,%d,%.9g,%.9g,%s,%d,%d\n", m.n, m.nt,
                m.nb, sym ? "symmetrized" : "original",
                storage ? "CSR" : "Dense", names[method],
                method == 0 ? omega : 0, r.it, r.sec, r.res, r.status, a.nnz,
                reps);
        fflush(f);
        fprintf(stderr, "%s %s: %d %.6fs residual %.3g %s\n",
                storage ? "CSR" : "Dense", names[method], r.it, r.sec, r.res,
                r.status);
        if (storage && method == 3) {
          snprintf(path, sizeof(path), "%s/solution%s_%d.csv", argv[2],
                   sym ? "s" : "", m.n);
          FILE *sf = fopen(path, "w");
          fprintf(sf, "x,y,T\n");
          for (int i = 0; i < m.n; i++)
            fprintf(sf, "%.17g,%.17g,%.17g\n", m.xy[2 * i], m.xy[2 * i + 1],
                    x[i]);
          fclose(sf);
        }
      }
    }
  }
  fclose(f);
  fclose(tf);
  free(x);
  free(b);
  free(a.d);
  free(a.rp);
  free(a.ci);
  free(a.v);
  free(a.diag);
  free(m.xy);
  free(m.g);
  free(m.tri);
  free(m.boundary);
  return failed ? 2 : 0;
}
