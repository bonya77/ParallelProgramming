#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <math.h>
#include <omp.h>
using namespace std;

const long N = 40000; 
const double EPS = 1e-5;  
const double TAU = 1e-5;  

void init(vector<double>& A, vector<double>& b, vector<double>& x) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; i++) {
        b[i] = N + 1.0;
        x[i] = 0.0;
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (i == j) ? 2.0 : 1.0; 
        }
    }
}

void solve_sequential(const vector<double>& A, const vector<double>& b, vector<double>& x) {
    vector<double> x_new(N, 0.0);
    double err = EPS + 1.0;
    
    while (err > EPS) {

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
            double Ax = 0.0;  
            for (int j = 0; j < N; j++) {
                Ax += A[i * N + j] * x[j];
            }
            x_new[i] = x[i] - TAU * (Ax - b[i]);  
        }

        err = 0.0;
        #pragma omp parallel for reduction(max:err) schedule(static)
        for (int i = 0; i < N; i++) {
            double diff = abs(x_new[i] - x[i]);
            if (diff > err) err = diff;
            x[i] = x_new[i]; 
        }
        
    }
}

int main() {
    vector<double> A(N * N), b(N), x(N);
    omp_set_num_threads(1);
    init(A, b, x);
    
    auto start = chrono::high_resolution_clock::now();
    solve_sequential(A, b, x);
    auto end = chrono::high_resolution_clock::now();
    
    chrono::duration<double> elapsed = end - start;
    cout << "Time Sequential: " << elapsed.count() << " s\n";
    
    cout << "First 5 components of solution:\n";
    for (int i = 0; i < std::min<double>(5, N); i++) {
        cout << "x[" << i << "] = " << x[i] << endl;
    }
    
    return 0;
}