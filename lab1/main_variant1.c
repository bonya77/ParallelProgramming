// main_variant1.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "solver.h"

#ifdef MPE_LOGGING
#include <mpe.h>

int event_start_compute, event_end_compute;
int event_start_comm, event_end_comm;
int event_iteration;
#endif

void print_header(int N, double eps, double tau, int size);
void print_result(const Result *result, int size);
void print_solution_sample(double *x, int N, int rank, const char *variant);

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int N = 2000;
    double eps = 1e-5;
    double tau = 1e-5;
    int max_iter = 50000;
    
    if (argc > 1) N = atoi(argv[1]);
    if (argc > 2) eps = atof(argv[2]);
    if (argc > 3) tau = atof(argv[3]);
    if (argc > 4) max_iter = atoi(argv[4]);

    #ifdef MPE_LOGGING
    MPE_Init_log();
    
    event_start_compute = MPE_Log_get_event_number();
    event_end_compute = MPE_Log_get_event_number();
    event_start_comm = MPE_Log_get_event_number();
    event_end_comm = MPE_Log_get_event_number();
    event_iteration = MPE_Log_get_event_number();
    
    MPE_Describe_state(event_start_compute, event_end_compute, "Compute", "red");
    MPE_Describe_state(event_start_comm, event_end_comm, "Communication", "blue");
    MPE_Describe_event(event_iteration, "Iteration", "green");
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPE_SetLogFileName("mpe_variant1");
    MPE_Start_log();
    #endif
    
    if (rank == 0) {
        print_header(N, eps, tau, size);
        printf("Running VARIANT 1 (full vectors)\n");
    }
    
    Result result = solve_variant1(N, eps, max_iter, tau, rank, size, 0.0);
    
    if (rank == 0) {
        print_solution_sample(result.solution, N, rank, "Variant 1");
        free(result.solution);
    }
    
    #ifdef MPE_LOGGING
    MPE_Finish_log("mpe_variant1");
    #endif
    
    if (rank == 0) {
        printf("\n==================================================\n");
        printf("RESULTS FOR VARIANT 1\n");
        printf("==================================================\n");
        print_result(&result, size);
        printf("==================================================\n");
        printf("Profile log saved to: mpe_variant1.clog2\n");
    }
    
    MPI_Finalize();
    return 0;
}

void print_header(int N, double eps, double tau, int size) {
    printf("==================================================\n");
    printf("VARIANT 1: Simple Iteration with Full Vectors\n");
    printf("==================================================\n");
    printf("System size:          %d x %d\n", N, N);
    printf("Number of processes:  %d\n", size);
    printf("Tolerance (epsilon):  %.1e\n", eps);
    if (tau > 0) {
        printf("Tau parameter:        %.6f\n", tau);
    } else {
        printf("Tau parameter:        auto\n");
    }
    printf("==================================================\n\n");
}

void print_result(const Result *result, int size) {
    printf("  Execution time:     %.3f seconds\n", result->time);
    printf("  Iterations:         %d\n", result->iterations);
    printf("  Final residual:     %.6e\n", result->residual);
    if (size > 1) {
        printf("  Speedup:            %.2f\n", result->speedup);
        printf("  Efficiency:         %.1f%%\n", result->efficiency);
    }
}