// solver.h
#ifndef SOLVER_H
#define SOLVER_H

#include <mpi.h>

typedef struct {
    int local_rows;
    int row_start;
    int *recvcounts;
    int *displs;
    int N;
    int rank;
    int size;
} DistInfo;

typedef struct {
    double time;
    int iterations;
    double residual;
    double speedup;
    double efficiency;
    double *solution;
} Result;

void init_distribution(int N, int rank, int size, DistInfo *info);
void free_distribution(DistInfo *info);
void generate_matrix_part(double *A_local, const DistInfo *info);
void generate_rhs_vector(double *b, const DistInfo *info);
void generate_rhs_local(double *b_local, const DistInfo *info);
double compute_residual_norm(double *Ax_local, double *b_local, 
                            const DistInfo *info, double b_norm);
double compute_b_norm(double *b_local, const DistInfo *info);
void matvec_local(double *A_local, double *x, double *result_local,
                  const DistInfo *info);
void matvec_local_full(double *A_local, double *x_full, double *result_local,
                       const DistInfo *info);
void print_solution_sample(double *x, int N, int rank, const char *variant);

Result solve_variant1(int N, double eps, int max_iter, double tau,
                     int rank, int size, double base_time);

#endif