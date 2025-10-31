// with omp for multiprocessing

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <omp.h>
#include <iostream>

inline float sinsum (double x, int terms){
    // sin(x) = x - x^3/3! + x^5/5! ... using the taylor series
    //first term of the series
    double term = x;
    // sum at 1
    double sum = term;
    // second term
    double x2 = x*x;

    for (int n = 1; n < terms; n++){
        // compute next term
        term *= -x2 / (double) (2*n*(2*n+1));
        sum += term;
    }
    return sum;
}

int main( int argc, char *argv[]){
    long long steps = argc > 1 ? atoll(argv[1]) : 1000000; // atoll for long long
    int terms = argc > 2 ? atoi(argv[2]) : 10; // atoi for int ie casts char* to int
    int threads = argc > 3 ? atoi(argv[3]) : 4; // number of threads

    double step_size = M_PI / (double) (steps -1);

    auto start = std::chrono::high_resolution_clock::now();

    double cpu_sum = 0.0;

    omp_set_num_threads(threads);
    #pragma omp parallel for reduction(+:cpu_sum)

    for (int step = 0; step < steps; step++){
        double x = step * step_size;
        cpu_sum += sinsum(x, terms);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;


    //trapezoidal rule correction
    cpu_sum -= 0.5 *(sinsum(0.0,terms)+sinsum(M_PI, terms));
    cpu_sum *= step_size;

    printf("cpu sum = %.10f,steps %lld terms %d time %.3f ms\n", cpu_sum, steps, terms, elapsed.count()*1000.0);

    return 0;


}
