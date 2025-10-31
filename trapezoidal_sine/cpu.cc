#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <iostream>

inline float sinsum (float x, int terms){
    // sin(x) = x - x^3/3! + x^5/5! ... using the taylor series
    //first term of the series
    float term = x;
    // sum at 1
    float sum = term;
    // second term
    float x2 = x*x;

    for (int n = 1; n < terms; n++){
        // compute next term
        term *= -x2 / (float) (2*n*(2*n+1));
        sum += term;
    }
    return sum;
}

int main( int argc, char *argv[]){
    long long steps = argc > 1 ? atoll(argv[1]) : 1000000; // atoll for long long
    int terms = argc > 2 ? atoi(argv[2]) : 10; // atoi for int ie casts char* to int

    double step_size = M_PI / (double) (steps -1);

    auto start = std::chrono::high_resolution_clock::now();

    double cpu_sum = 0.0;

    for (int step = 0; step < steps; step++){
        float x = step * step_size;
        cpu_sum += sinsum(x, terms);
    }
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Elapsed time: " << elapsed.count() << " seconds" << std::endl;


    //trapezoidal rule correction
    cpu_sum -= 0.5 *(sinsum(0.0,terms)+sinsum(M_PI, terms));
    cpu_sum *= step_size;

    printf("cpu sum = %.10f,steps %d terms %d time %.3f ms\n", cpu_sum, steps, terms, elapsed.count()*1000.0);

    return 0;


}
