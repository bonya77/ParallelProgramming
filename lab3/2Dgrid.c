#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

double valA(int i, int j) {
  return sin(i * 0.01) + cos(j * 0.02) + (i * j % 7) * 0.1;
}
double valB(int i, int j) {
  return cos(i * 0.03) + sin(j * 0.04) + (i + j % 5) * 0.2;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n1 = atoi(argv[1]);
  int n2 = atoi(argv[2]);
  int n3 = atoi(argv[3]);

  int dims[2] = {0, 0};
  MPI_Dims_create(size, 2, dims);
  int px = dims[0], py = dims[1];

  int periods[2] = {0, 0};
  MPI_Comm grid_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);

  int coords[2];
  MPI_Cart_coords(grid_comm, rank, 2, coords);

  if (n1 % px != 0 || n3 % py != 0) {
    if (rank == 0) printf("Error: n1 must be divisible by px and n3 by py\n");
    MPI_Finalize();
    return 1;
  }

  int rows = n1 / px;
  int cols = n3 / py;

  MPI_Comm row_comm, col_comm;
  int remain_dims[2];

  remain_dims[0] = 0;
  remain_dims[1] = 1;
  MPI_Cart_sub(grid_comm, remain_dims, &row_comm);

  remain_dims[0] = 1;
  remain_dims[1] = 0;
  MPI_Cart_sub(grid_comm, remain_dims, &col_comm);

  double *A = NULL, *B = NULL, *C = NULL;
  double *Ablock = malloc(rows * n2 * sizeof(double));
  double *Bblock = malloc(n2 * cols * sizeof(double));
  double *Cblock = calloc(rows * cols, sizeof(double));

  MPI_Datatype col_type, res_type;
  double start = MPI_Wtime();
  if (rank == 0) {
    A = malloc(n1 * n2 * sizeof(double));
    B = malloc(n2 * n3 * sizeof(double));
    C = malloc(n1 * n3 * sizeof(double));

    for (int i = 0; i < n1; i++)
      for (int j = 0; j < n2; j++) A[i * n2 + j] = valA(i, j);
    for (int i = 0; i < n2; i++)
      for (int j = 0; j < n3; j++) B[i * n3 + j] = valB(i, j);

    MPI_Type_vector(n2, cols, n3, MPI_DOUBLE, &col_type);
    MPI_Type_create_resized(col_type, 0, cols * sizeof(double), &col_type);
    MPI_Type_commit(&col_type);

    MPI_Type_vector(rows, cols, n3, MPI_DOUBLE, &res_type);
    MPI_Type_create_resized(res_type, 0, sizeof(double), &res_type);
    MPI_Type_commit(&res_type);
  }

if (coords[1] == 0) {
    MPI_Scatter(A, rows * n2, MPI_DOUBLE, Ablock, rows * n2, MPI_DOUBLE, 0,
                col_comm);
  }

  MPI_Bcast(Ablock, rows * n2, MPI_DOUBLE, 0, row_comm);

  if (coords[0] == 0) {
    MPI_Scatter(B, 1, col_type, Bblock, n2 * cols, MPI_DOUBLE, 0, row_comm);
  }
  MPI_Bcast(Bblock, n2 * cols, MPI_DOUBLE, 0, col_comm);

  for (int i = 0; i < rows; i++) {
    for (int k = 0; k < n2; k++) {
      double a_ik = Ablock[i * n2 + k];
      for (int j = 0; j < cols; j++) {
        Cblock[i * cols + j] += a_ik * Bblock[k * cols + j];
      }
    }
  }

  int *sendcounts = NULL, *displs = NULL;
  if (rank == 0) {
    sendcounts = malloc(size * sizeof(int));
    displs = malloc(size * sizeof(int));
    for (int p = 0; p < size; p++) {
      int p_coords[2];
      MPI_Cart_coords(grid_comm, p, 2, p_coords);
      sendcounts[p] = 1;
      displs[p] = (p_coords[0] * rows * n3) + (p_coords[1] * cols);
    }
  }

  MPI_Gatherv(Cblock, rows * cols, MPI_DOUBLE, C, sendcounts, displs, res_type,
              0, grid_comm);
  double end = MPI_Wtime();
  if (rank == 0) {
    printf("Matrix %dx%d x %dx%d completed.\n", n1, n2, n3);
    printf("Time: %f sec\n", end - start);

    free(A);
    free(B);
    free(C);
    free(sendcounts);
    free(displs);
    MPI_Type_free(&col_type);
    MPI_Type_free(&res_type);
  }

  free(Ablock);
  free(Bblock);
  free(Cblock);
  MPI_Finalize();
  return 0;
}