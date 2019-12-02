#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//constants
#define n 0.0002
#define p 0.5
#define G 0.75
//parameters
#define SIZE 4
#define NUMBER_OF_ITERATIONS 3
#define ALGORITHM 4
//1 = sequential
//2 = simple parallel
//3 = 1 block N threads
//4 = 1 N blocks 1 thread

#define DEBUG 1
//if DEBUG, there will be printf msgs

__device__ int idx(int i, int j){
    return (SIZE * i + j);
}

int idx_seq(int i, int j){
    return (SIZE * i + j);
}

__device__ void update(double * u, double * u1, double *u2, int i, int j){
    //taken care of by other threads
    if(i == 0 || j == 0 || i == SIZE-1 || j == SIZE-1){
        return;
    }
    //middle elements
    u[idx(i, j)]=  p * 
                            (u1[idx(i-1,j)] + u1[idx(i+1,j)] 
                            +u1[idx(i,j-1)] + u1[idx(i,j+1)] 
                        - 4 * u1[idx(i, j)])  
                        + 2 * u1[idx(i, j)] - (1-n) * u2[idx(i, j)];
    //sides & corners merged
    if(j==1){
        u[idx(i,0)] = G * u[idx(i, j)];
        //top left corner
        if(i == 1){
            u[idx(0,0)] = G * u[idx(1,0)];
        }
        //top right corner
        if(i == SIZE-2){
            u[idx(SIZE-1,0)] = G * u[idx(SIZE-2, 0)];
        }
    }
    if(i==1){
        u[idx(0, j)] = G * u[idx(i, j)];
        //bottom left corner
        if(j==SIZE-2){
            u[idx(0,SIZE-1)] = G * u[idx(0, SIZE-2)];
        }
    }
    if(j == SIZE-2){
        u[idx(i, SIZE-1)]  = G * u[idx(i, j)];
    }
    if(i == SIZE-2){
        u[idx(SIZE-1, j)]  = G * u[idx(i, j)];
        //bottom right corner
        if(j== SIZE-2){
            u[idx(SIZE-1, SIZE-1)] = G * u[idx(SIZE-1, SIZE-2)];
        }
    }
}

__global__ void updateElementThree(double *u, double *u1, double *u2)
{
    int j = threadIdx.x;
    for(int i=0; i < SIZE; i++)
    {
        update(u, u1, u2, i, j);
    }
}

__global__ void updateElementFour(double *u, double *u1, double *u2)
{
    int i = blockIdx.x;  
    for(int j=0; j < SIZE; j++)
    {
        update(u, u1, u2, i, j);
    }
}

__global__ void updateElementTwo(double *u, double *u1, double *u2)
{
    int i = blockIdx.x;  
    int j = threadIdx.x;
    update(u, u1, u2, i, j);
}

void printMatrix(double* u){
        printf("\n");
        for(int i = 0; i < SIZE * SIZE; i++){
            printf("%.3lf", u[i]);
            printf("\t");
            if((i+1) %  SIZE == 0 && i > 0){
                printf("\n");
            }
        }
}

int main(){
    //Code is in C, but the nvcc compiler expects C++
    //C likes it when you implicitly cast a void* to a *double
    //but C++ really doesnt like it when you implicitly cast a void* to a double*
    double* u  = static_cast<double*>(malloc(sizeof(double) * SIZE * SIZE ));
    double* u1 = static_cast<double*>(malloc(sizeof(double) * SIZE * SIZE ));
    double* u2 = static_cast<double*>(malloc(sizeof(double) * SIZE * SIZE ));

    //initialize to 0
    for(int i = 0; i < SIZE * SIZE; i++){
        u1[i] = 0;
        u2[i] = 0;
    }

    //hit that drummmm
    u1[(SIZE * SIZE/2 + SIZE/2)] = 1;

    clock_t start, end;
    double cpu_time_used;
    if(ALGORITHM == 1)
    {
        for(int a=0; a < NUMBER_OF_ITERATIONS; a++)
        {    
            for(int i = 1; i <SIZE-1; i++){
                for(int j = 1; j <SIZE-1 ; j++){
                //middle elements
                u[idx_seq(i, j)]=  p * 
                                        (u1[idx_seq(i-1,j)] + u1[idx_seq(i+1,j)] 
                                        +u1[idx_seq(i,j-1)] + u1[idx_seq(i,j+1)] 
                                        - 4 * u1[idx_seq(i, j)])  
                                    + 2 * u1[idx_seq(i, j)] - (1-n) * u2[idx_seq(i, j)];
                }
            }
            //sides
            for(int i = 1; i < SIZE-1; i++){
                u[idx_seq(0, i)]       = G * u[idx_seq(1, i)];
                u[idx_seq(SIZE-1, i)]  = G * u[idx_seq(SIZE-2, i)];
                u[idx_seq(i,0)]        = G * u[idx_seq(i, 1)];
                u[idx_seq(i, SIZE-1)]  = G * u[idx_seq(i, SIZE-2)];
            }
            //corners
            u[idx_seq(0,0)]            = G * u[idx_seq(1,0)];
            u[idx_seq(SIZE-1,0)]       = G * u[idx_seq(SIZE-2, 0)];
            u[idx_seq(0,SIZE-1)]       = G * u[idx_seq(0, SIZE-2)];
            u[idx_seq(SIZE-1, SIZE-1)] = G * u[idx_seq(SIZE-1, SIZE-2)];
            //update after iterations
            for(int i=0; i < SIZE * SIZE; i++){
                u2[i] = u1[i];
                u1[i] = u[i];
            }
            if(DEBUG){
                if(SIZE > 4){
                    printf("\n%lf", u[(SIZE * SIZE/2 + SIZE/2)] );
                } else {
                    printMatrix(u);
                }
            } 
        }
    }
    else
    {   
        double* u_dev, *u2_dev; 
        double *u1_dev;
        cudaMalloc((void **)&u_dev, SIZE*SIZE *sizeof(double));
        cudaMalloc((void **)&u1_dev, SIZE*SIZE *sizeof(double));
        cudaMalloc((void **)&u2_dev, SIZE*SIZE *sizeof(double));  
    
        cudaMemcpy(u_dev, u, SIZE*SIZE *sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(u1_dev, u1, SIZE*SIZE *sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(u2_dev, u2, SIZE*SIZE *sizeof(double), cudaMemcpyHostToDevice);
        //hit that dRuMm
        u1[(SIZE * SIZE/2 + SIZE/2)] = 1;
        start = clock();
        for(int i = 0; i < NUMBER_OF_ITERATIONS ; i++){
            if(ALGORITHM ==2){
                updateElementTwo << <SIZE, SIZE >> > (u_dev, u1_dev, u2_dev);
            }
            if(ALGORITHM ==3 ){
                updateElementThree << <1, SIZE >> > (u_dev, u1_dev, u2_dev);
            }
            if(ALGORITHM == 4){
                updateElementFour << <SIZE, 1 >> > (u_dev, u1_dev, u2_dev);
            }
            cudaDeviceSynchronize();
            if(DEBUG){
                cudaMemcpy(u, u_dev, SIZE*SIZE *sizeof(double), cudaMemcpyDeviceToHost);
                if(SIZE > 4){
                    printf("\n%lf", u[(SIZE * SIZE/2 + SIZE/2)] );
                } else {
                    printMatrix(u);
                }
            }
            //ideally, replace this two statements by a
            //__global__ function that updates the matrices in parallel
            cudaMemcpy(u2_dev, u1_dev, SIZE*SIZE *sizeof(double), cudaMemcpyDeviceToDevice);
            cudaMemcpy(u1_dev, u_dev, SIZE*SIZE *sizeof(double), cudaMemcpyDeviceToDevice);
        }
        cudaFree(u_dev);
        cudaFree(u1_dev);
        cudaFree(u2_dev);
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("\n\nAlgorithm: ");
    switch (ALGORITHM){
        case 1:
            printf("Sequential\n");
            break;
        case 2:
            printf("Simple parallel\n");
            break;
        case 3:
            printf("1 block with N threads\n");
            break;
        case 4:
            printf("N blocks with 1 thread\n");
            break;
        default: break;
    }
    printf("\nExecution time:\t%lf \n\n", cpu_time_used);
    free(u);
    free(u1);
    free(u2);
}