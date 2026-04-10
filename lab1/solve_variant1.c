// solve_variant1.c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "solver.h"

#ifdef MPE_LOGGING
#include <mpe.h>

extern int event_start_compute, event_end_compute;
extern int event_start_comm, event_end_comm;
extern int event_iteration;
#endif

Result solve_variant1(int N, double eps, int max_iter, double tau,
                     int rank, int size, double base_time) {
    Result result = {0.0, 0, 0.0, 1.0, 100.0, NULL};
    
    DistInfo info;
    init_distribution(N, rank, size, &info);
    
    if (tau <= 0) {
        tau = 0.9 / (N + 1);
        if (rank == 0) {
            printf("V1: Using auto tau = %.6f\n", tau);
        }
    }
    
    double *A_local = malloc(info.local_rows * N * sizeof(double));
    double *x = malloc(N * sizeof(double));
    double *b = malloc(N * sizeof(double));
    double *Ax_local = malloc(info.local_rows * sizeof(double));
    double *b_local = malloc(info.local_rows * sizeof(double));
    
    if (!A_local || !x || !b || !Ax_local || !b_local) {
        if (rank == 0) printf("V1: Memory allocation failed!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    generate_matrix_part(A_local, &info);
    generate_rhs_vector(b, &info);
    
    for (int i = 0; i < info.local_rows; i++) {
        b_local[i] = b[info.row_start + i];
    }
    
    for (int i = 0; i < N; i++) {
        x[i] = 0.0;
    }
    
    double b_norm = compute_b_norm(b_local, &info);
    
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    
    int iter;
    double residual_norm = 1.0;
    
    for (iter = 0; iter < max_iter; iter++) {
        //if (rank == 1 && iter == 10) {
        //    printf("Rank 1: Выхожу из цикла досрочно на итерации %d!\n", iter);
         //   break;
        //}
        #ifdef MPE_LOGGING
        MPE_Log_event(event_iteration, rank, "Iteration");
        MPE_Log_event(event_start_compute, rank, "Start compute");
        #endif
        
        matvec_local(A_local, x, Ax_local, &info);
        
        #ifdef MPE_LOGGING
        MPE_Log_event(event_end_compute, rank, "End compute");
        MPE_Log_event(event_start_comm, rank, "Start comm");
        #endif
        
        residual_norm = compute_residual_norm(Ax_local, b_local, &info, b_norm);
        
        #ifdef MPE_LOGGING
        MPE_Log_event(event_end_comm, rank, "End comm");
        #endif
        
        if (residual_norm < eps) {
            break;
        }
        
        for (int i = 0; i < info.local_rows; i++) {
            int global_idx = info.row_start + i;
            x[global_idx] -= tau * (Ax_local[i] - b_local[i]);
        }
        
        #ifdef MPE_LOGGING
        MPE_Log_event(event_start_comm, rank, "Start comm");
        #endif
        
        MPI_Allgatherv(MPI_IN_PLACE, info.local_rows, MPI_DOUBLE,
                      x, info.recvcounts, info.displs, MPI_DOUBLE, MPI_COMM_WORLD);
        
        #ifdef MPE_LOGGING
        MPE_Log_event(event_end_comm, rank, "End comm");
        #endif
        
        if (rank == 0 && iter % 1000 == 0) {
            printf("V1: Iteration %d, residual = %.6e\n", iter, residual_norm);
        }
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();
    
    result.time = end_time - start_time;
    result.iterations = iter;
    result.residual = residual_norm;
    
    if (base_time > 0) {
        result.speedup = base_time / result.time;
        result.efficiency = (result.speedup / size) * 100.0;
    }
    
    // Сохраняем решение в результате
    if (rank == 0) {
        result.solution = malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) {
            result.solution[i] = x[i];
        }
    }
    
    free(A_local);
    free(x);
    free(b);
    free(Ax_local);
    free(b_local);
    free_distribution(&info);
    
    return result;
}