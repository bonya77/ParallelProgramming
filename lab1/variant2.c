// variant2.c
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define TAU 0.01

void print_solution_sample_v2(double *x, int N, int rank, int *counts, int *displs) {
    int sample_size = (N < 10) ? N : 10;
    double *x_full = NULL;
    
    if (rank == 0) {
        x_full = malloc(N * sizeof(double));
    }
    
    MPI_Gatherv(x, counts[rank], MPI_DOUBLE,
                x_full, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("\nVariant 2 - First %d elements of solution:\n", sample_size);
        printf("[");
        for (int i = 0; i < sample_size; i++) {
            printf("%.6f", x_full[i]);
            if (i < sample_size - 1) printf(", ");
        }
        printf("]\n");
        
        double exact = 1.0;
        double max_diff = 0.0;
        for (int i = 0; i < sample_size; i++) {
            double diff = fabs(x_full[i] - exact);
            if (diff > max_diff) max_diff = diff;
        }
        printf("Max difference from exact solution (1.0): %.2e\n", max_diff);
        
        if (max_diff < 1e-4) {
            printf("✓ Solution verification PASSED\n");
        } else {
            printf("✗ Solution verification FAILED\n");
        }
        
        free(x_full);
    }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int N = 2000;
  double eps = 1e-5;

  if (argc > 1) N = atoi(argv[1]);
  if (argc > 2) eps = atof(argv[2]);

  int base = N / size;
  int rem = N % size;
  int rows = base + (rank < rem ? 1 : 0);

  int *counts = malloc(size * sizeof(int));
  int *displs = malloc(size * sizeof(int));

  for (int i = 0; i < size; i++) {
    counts[i] = base + (i < rem ? 1 : 0);
    displs[i] = (i == 0 ? 0 : displs[i - 1] + counts[i - 1]);
  }

  int start = displs[rank];

  double *x_loc = malloc(rows * sizeof(double));
  double *b_loc = malloc(rows * sizeof(double));
  double *Ax_loc = malloc(rows * sizeof(double));

  int max_rows = base + 1;
  double *x_buf = malloc(max_rows * sizeof(double));

  for (int i = 0; i < rows; i++) {
    x_loc[i] = 0.0;
    b_loc[i] = N + 1.0;
  }

  double err = 1.0;
  double t1 = MPI_Wtime();
  int iter = 0;

  while (err > eps) {
    iter++;

    for (int i = 0; i < rows; i++)
      Ax_loc[i] = 0.0;

    int owner = rank;
    int cur_count = rows;

    for (int i = 0; i < cur_count; i++)
      x_buf[i] = x_loc[i];

    for (int step = 0; step < size; step++) {

      int global_start = displs[owner];

      for (int i = 0; i < rows; i++) {
        int gi = start + i;

        for (int j = 0; j < cur_count; j++) {
          int gj = global_start + j;

          if (gi == gj)
            Ax_loc[i] += 2.0 * x_buf[j];
          else
            Ax_loc[i] += 1.0 * x_buf[j];
        }
      }

      int send_to = (rank + 1) % size;
      int recv_from = (rank - 1 + size) % size;

      int next_owner = (owner - 1 + size) % size;
      int next_count = counts[next_owner];

      MPI_Sendrecv_replace(x_buf, max_rows, MPI_DOUBLE,
                           send_to, 0,
                           recv_from, 0,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      owner = next_owner;
      cur_count = next_count;
    }

    double err_loc = 0.0;

    for (int i = 0; i < rows; i++) {
      double r = Ax_loc[i] - b_loc[i];
      err_loc += r * r;
      x_loc[i] -= TAU * r;
    }

    MPI_Allreduce(&err_loc, &err, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);

    err = sqrt(err);
    
    if (rank == 0 && iter % 1000 == 0) {
        printf("V2: Iteration %d, residual = %.6e\n", iter, err);
    }
  }

  double t2 = MPI_Wtime();

  if (rank == 0) {
    printf("\n==================================================\n");
    printf("VARIANT 2: Ring Pipeline Implementation\n");
    printf("==================================================\n");
    printf("System size:          %d x %d\n", N, N);
    printf("Number of processes:  %d\n", size);
    printf("Tolerance (epsilon):  %.1e\n", eps);
    printf("Tau parameter:        %.6f\n", TAU);
    printf("==================================================\n");
    printf("Time (Ring):          %lf seconds\n", t2 - t1);
    printf("Iterations:           %d\n", iter);
    printf("Final residual:       %.6e\n", err);
  }
  
  print_solution_sample_v2(x_loc, N, rank, counts, displs);

  free(x_loc);
  free(b_loc);
  free(Ax_loc);
  free(x_buf);
  free(counts);
  free(displs);

  MPI_Finalize();
  return 0;
}