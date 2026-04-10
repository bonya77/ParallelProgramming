#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "solver.h"

void init_distribution(int N, int rank, int size, DistInfo *info) {
    info->N = N;
    info->rank = rank;
    info->size = size;
    
    info->local_rows = N / size;
    int remainder = N % size;
    
    if (rank < remainder) {
        info->local_rows++;
    }
    
    info->row_start = 0;
    for (int i = 0; i < rank; i++) {
        int rows_i = N / size + (i < remainder ? 1 : 0);
        info->row_start += rows_i;
    }
    
    info->recvcounts = malloc(size * sizeof(int));
    info->displs = malloc(size * sizeof(int));
    
    int offset = 0;
    for (int i = 0; i < size; i++) {
        int rows_i = N / size + (i < remainder ? 1 : 0);
        info->recvcounts[i] = rows_i;
        info->displs[i] = offset;
        offset += rows_i;
    }
}

void free_distribution(DistInfo *info) {
    free(info->recvcounts);
    free(info->displs);
}

void generate_matrix_part(double *A_local, const DistInfo *info) {
    int N = info->N;
    for (int i = 0; i < info->local_rows; i++) {
        int global_row = info->row_start + i;
        for (int j = 0; j < N; j++) {
            if (global_row == j) {
                A_local[i * N + j] = 2.0;
            } else {
                A_local[i * N + j] = 1.0;
            }
        }
    }
}

void generate_rhs_vector(double *b, const DistInfo *info) {
    for (int i = 0; i < info->N; i++) {
        b[i] = info->N + 1.0;
    }
}

void generate_rhs_local(double *b_local, const DistInfo *info) {
    for (int i = 0; i < info->local_rows; i++) {
        b_local[i] = info->N + 1.0;
    }
}

void matvec_local(double *A_local, double *x, double *result_local,
                  const DistInfo *info) {
    int N = info->N;
    for (int i = 0; i < info->local_rows; i++) {
        result_local[i] = 0.0;
        for (int j = 0; j < N; j++) {
            result_local[i] += A_local[i * N + j] * x[j];
        }
    }
}

void matvec_local_full(double *A_local, double *x_full, double *result_local,
                       const DistInfo *info) {
    int N = info->N;
    for (int i = 0; i < info->local_rows; i++) {
        result_local[i] = 0.0;
        for (int j = 0; j < N; j++) {
            result_local[i] += A_local[i * N + j] * x_full[j];
        }
    }
}

double compute_residual_norm(double *Ax_local, double *b_local,
                            const DistInfo *info, double b_norm) {
    double local_sum = 0.0;
    for (int i = 0; i < info->local_rows; i++) {
        double diff = Ax_local[i] - b_local[i];
        if (!isfinite(diff)) {
            return INFINITY;
        }
        local_sum += diff * diff;
    }
    
    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    if (global_sum < 0) global_sum = 0;
    
    return sqrt(global_sum) / b_norm;
}

double compute_b_norm(double *b_local, const DistInfo *info) {
    double local_sum = 0.0;
    for (int i = 0; i < info->local_rows; i++) {
        local_sum += b_local[i] * b_local[i];
    }
    
    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    return sqrt(global_sum);
}

void print_solution_sample(double *x, int N, int rank, const char *variant) {
    int sample_size = (N < 10) ? N : 10;
    
    if (rank == 0) {
        printf("\n%s - First %d elements of solution:\n", variant, sample_size);
        printf("[");
        for (int i = 0; i < sample_size; i++) {
            printf("%.6f", x[i]);
            if (i < sample_size - 1) printf(", ");
        }
        printf("]\n");
        
        double exact = 1.0;
        double max_diff = 0.0;
        for (int i = 0; i < sample_size; i++) {
            double diff = fabs(x[i] - exact);
            if (diff > max_diff) max_diff = diff;
        }
        printf("Max difference from exact solution (1.0): %.2e\n", max_diff);
        
        if (max_diff < 1e-4) {
            printf("✓ Solution verification PASSED\n");
        } else {
            printf("✗ Solution verification FAILED\n");
        }
    }
}