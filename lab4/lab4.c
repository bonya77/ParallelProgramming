#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define IDX(i, j, k) ((i) * ny * nx + (j) * nx + (k))

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);


  int N = atoi(argv[1]);
  int nx = N, ny = N, nz = N;


  double Dx = 2.0, Dy = 2.0, Dz = 2.0; 
  double x0 = -1.0, y0 = -1.0, z0 = -1.0;
  double hx = Dx / (nx - 1);
  double hy = Dy / (ny - 1);
  double hz = Dz / (nz - 1);
  double a = 1e5;
  double eps = 1e-8;

  int rows = nz / size;
  int rest = nz % size;
  int local_nz = rows + (rank < rest);  
  int top = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
  int bottom = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

  int height = local_nz + 2;
  int slice = ny * nx; 

  double *phi = calloc(height * slice, sizeof(double));
  double *phi_new = calloc(height * slice, sizeof(double));

  int z_offset = rank * rows + (rank < rest ? rank : rest);

  for (int i = 1; i <= local_nz; i++) {
    int k_global = z_offset + (i - 1);
    double zk = z0 + k_global * hz;

    for (int j = 0; j < ny; j++) {
      double yj = y0 + j * hy;
      for (int k = 0; k < nx; k++) {
        double xk = x0 + k * hx;

        int on_boundary = (k_global == 0 || k_global == nz - 1 || j == 0 ||
                           j == ny - 1 || k == 0 || k == nx - 1);
        if (on_boundary) {
          double val = xk * xk + yj * yj + zk * zk;
          phi[IDX(i, j, k)] = val;
          phi_new[IDX(i, j, k)] = val;  
        }
      }
    }
  }

  double denom = 2.0 / (hx * hx) + 2.0 / (hy * hy) + 2.0 / (hz * hz) + a;

  double start = MPI_Wtime();

  double global_diff = 1.0;
  int iter = 0;
  int iter_max = 10000;

  while (global_diff > eps && iter < iter_max) {


    MPI_Request req[4];

    MPI_Irecv(&phi[IDX(0, 0, 0)], slice, MPI_DOUBLE, top, 0, MPI_COMM_WORLD,
              &req[0]);
    MPI_Irecv(&phi[IDX(local_nz + 1, 0, 0)], slice, MPI_DOUBLE, bottom, 0,
              MPI_COMM_WORLD, &req[1]);

    MPI_Isend(&phi[IDX(1, 0, 0)], slice, MPI_DOUBLE, top, 0, MPI_COMM_WORLD,
              &req[2]);
    MPI_Isend(&phi[IDX(local_nz, 0, 0)], slice, MPI_DOUBLE, bottom, 0,
              MPI_COMM_WORLD, &req[3]);


    double local_diff = 0.0;

    for (int i = 2; i <= local_nz - 1; i++) {
      int k_global = z_offset + (i - 1);
      double zk = z0 + k_global * hz;

      for (int j = 1; j < ny - 1; j++) {
        double yj = y0 + j * hy;
        for (int k = 1; k < nx - 1; k++) {
          double xk = x0 + k * hx;

          double rho = 6.0 - a * (xk * xk + yj * yj + zk * zk);
          double nb =
              (phi[IDX(i + 1, j, k)] + phi[IDX(i - 1, j, k)]) / (hz * hz) +
              (phi[IDX(i, j + 1, k)] + phi[IDX(i, j - 1, k)]) / (hy * hy) +
              (phi[IDX(i, j, k + 1)] + phi[IDX(i, j, k - 1)]) / (hx * hx);

          phi_new[IDX(i, j, k)] = (nb - rho) / denom;

          double d = fabs(phi_new[IDX(i, j, k)] - phi[IDX(i, j, k)]);
          if (d > local_diff) local_diff = d;
        }
      }
    }


    MPI_Waitall(4, req, MPI_STATUSES_IGNORE);


    for (int edge = 0; edge < 2; edge++) {
      int i = (edge == 0) ? 1 : local_nz;
      int k_global = z_offset + (i - 1);
      double zk = z0 + k_global * hz;

      if (k_global == 0 || k_global == nz - 1) continue;

      for (int j = 1; j < ny - 1; j++) {
        double yj = y0 + j * hy;
        for (int k = 1; k < nx - 1; k++) {
          double xk = x0 + k * hx;

          double rho = 6.0 - a * (xk * xk + yj * yj + zk * zk);

          double nb =
              (phi[IDX(i + 1, j, k)] + phi[IDX(i - 1, j, k)]) / (hz * hz) +
              (phi[IDX(i, j + 1, k)] + phi[IDX(i, j - 1, k)]) / (hy * hy) +
              (phi[IDX(i, j, k + 1)] + phi[IDX(i, j, k - 1)]) / (hx * hx);

          phi_new[IDX(i, j, k)] = (nb - rho) / denom;

          double d = fabs(phi_new[IDX(i, j, k)] - phi[IDX(i, j, k)]);
          if (d > local_diff) local_diff = d;
        }
      }
    }
    


    MPI_Allreduce(&local_diff, &global_diff, 1, MPI_DOUBLE, MPI_MAX,
                  MPI_COMM_WORLD);

    double *tmp = phi;
    phi = phi_new;
    phi_new = tmp;

    iter++;
    if (rank == 0 && iter % 100 == 0)
      printf("iter %d, diff = %.2e\n", iter, global_diff);
  }

  double end = MPI_Wtime();
  double time = end - start;

  double local_err = 0.0;
  for (int i = 1; i <= local_nz; i++) {
    int k_global = z_offset + (i - 1);
    double zk = z0 + k_global * hz;
    for (int j = 0; j < ny; j++) {
      double yj = y0 + j * hy;
      for (int k = 0; k < nx; k++) {
        double xk = x0 + k * hx;
        double exact = xk * xk + yj * yj + zk * zk;
        double d = fabs(phi[IDX(i, j, k)] - exact);
        if (d > local_err) local_err = d;
      }
    }
  }

  double max_time, max_err;
  MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_err, &max_err, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0)
    printf("Iters: %d | Time: %.4f s | Max error: %.2e\n", iter, max_time,
           max_err);

  free(phi);
  free(phi_new);
  MPI_Finalize();
  return 0;
}