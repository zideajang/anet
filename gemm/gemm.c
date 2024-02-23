#include<stdint.h>
#include<time.h>
#include <immintrin.h> 
#include <stdio.h>


#define BLOCK 8
#define N 1024

float A[N*N] __attribute__ ((aligned (32)));
float B[N*N] __attribute__ ((aligned (32)));
float C[N*N] __attribute__ ((aligned (32)));
float val[N*N] __attribute__ ((aligned (32)));

__m256 *Am = (__m256*)A;
__m256 *Bm = (__m256*)B;
__m256 *Cm = (__m256*)C;

uint64_t nanos(){
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC_RAW , &start);
    return (uint64_t)start.tv_sec *1000000000 + (uint64_t)start.tv_nsec ;
}


void matmul()
{
    for (int by = 0; by < N; by += BLOCK){
        for (int bx = 0; bx < N; bx += BLOCK){
#ifndef FAST
    float tc[BLOCK][BLOCK];
    for (int y = 0; y < BLOCK; y++){
        for (int  x = 0; x < BLOCK; x++){
            float acc = 0;
            for (int k = 0; k < N; k++){
                acc += A[(by + y)*N + k] * B[(bx + x)*N + k];
            }
            tc[y][x] = acc;    
        }
    }

    //store
    for (int y = 0; y < BLOCK; y++)
    {
        for (int x = 0; x < BLOCK; x++)
        {
            C[(by + y)*N + bx + x] = tc[y][x];
        }
    }
#else
    float tc[BLOCK][BLOCK];
    for (int y = 0; y < BLOCK; y++)
    {
        for (int x = 0; x < BLOCK; x++)
        {
            __m256 tmp = {};
            for (int k = 0; k < N; k++)
            {
                tmp = _mm256_fmadd_ps(Am[((by + y)*N + k)/8],Am[((bx + x)*N + k)/8],tmp);
            }
            tc[y][x] = tmp;
            
        }
        
    }

    for (size_t i = 0; i < count; i++)
    {
        /* code */
    }
    
    

#endif


        }
        
    }
    
}


int main(void){
    printf("hello\n");

    assert(N%BLOCK == 0);

    uint64_t start = nanos();

    

    uint64_t end = nanos();
    // printf("%d\n",end);
    double gflops = (2.0*N*N*N)*1e-9;
    double s = (end - start)*1e-9;
    printf("%f GFLOP/s\n",gflops/s);
    
    return 0;
}
