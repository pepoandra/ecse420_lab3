#include <stdio.h>
#include <stdlib.h>
#include <time.h>


#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define n 0.0002
#define p 0.5
#define G 0.75

#define SIZE 1024
#define NUMBER_OF_ITERATIONS 100
#define DEBUG 0


__device__ int idx(int i, int j){
    return (SIZE * i + j);
}


__global__ void foo(double *u1){
    int i = blockIdx.x;  
    int j = threadIdx.x;
    printf("u1[%d,%d]: %.3lf  \t",i, j,  u1[idx(i,j)] );
    
}

__global__ void updateElement(double *u, double *u1, double *u2)
{
    int i = blockIdx.x;  
    int j = threadIdx.x;

    //printf("i: %d j: %d \n", i, j);

    for(int j=0; j < SIZE; j++)
    //for(int i=0; i < SIZE; i++)
    {
        //taken care of by other threads
        if(i == 0 || j == 0 || i == SIZE-1 || j == SIZE-1){
            continue;
        }

        u[idx(i, j)]=  p * 
                                (u1[idx(i-1,j)] + u1[idx(i+1,j)] 
                                +u1[idx(i,j-1)] + u1[idx(i,j+1)] 
                            - 4 * u1[idx(i, j)])  
                            + 2 * u1[idx(i, j)] - (1-n) * u2[idx(i, j)];

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

    double* u  = static_cast<double*>(malloc(sizeof(double) * SIZE * SIZE ));
    double* u1 = static_cast<double*>(malloc(sizeof(double) * SIZE * SIZE ));
    double* u2 = static_cast<double*>(malloc(sizeof(double) * SIZE * SIZE ));


    //initialize to 0
    for(int i = 0; i < SIZE * SIZE; i++){
        //u[i] = 0;
        u1[i] = 0;
        u2[i] = 0;
    }

    //hit that drummmm
    //u1[idx(SIZE/2, SIZE/2)] = 1;
    u1[(SIZE * SIZE/2 + SIZE/2)] = 1;
    //printMatrix(u1);

    clock_t start, end;
    double cpu_time_used;
    
    double* u_dev, *u2_dev; 
    double *u1_dev;
    cudaMalloc((void **)&u_dev, SIZE*SIZE *sizeof(double));
    cudaMalloc((void **)&u1_dev, SIZE*SIZE *sizeof(double));
    cudaMalloc((void **)&u2_dev, SIZE*SIZE *sizeof(double));  
  
    cudaMemcpy(u_dev, u, SIZE*SIZE *sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(u1_dev, u1, SIZE*SIZE *sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(u2_dev, u2, SIZE*SIZE *sizeof(double), cudaMemcpyHostToDevice);

    u1[(SIZE * SIZE/2 + SIZE/2)] = 1;

    start = clock();

    for(int i = 0; i < NUMBER_OF_ITERATIONS ; i++){
        updateElement << <SIZE, 1 >> > (u_dev, u1_dev, u2_dev);
        cudaDeviceSynchronize();
        
        if(DEBUG){
            cudaMemcpy(u, u_dev, SIZE*SIZE *sizeof(double), cudaMemcpyDeviceToHost);
            //cudaMemcpy(u1, u1_dev, SIZE*SIZE *sizeof(double), cudaMemcpyDeviceToHost);

            //printMatrix(u);
            printf("\n\n%lf", u[(SIZE * SIZE/2 + SIZE/2)] );
        }
        cudaMemcpy(u2_dev, u1_dev, SIZE*SIZE *sizeof(double), cudaMemcpyDeviceToDevice);
        cudaMemcpy(u1_dev, u_dev, SIZE*SIZE *sizeof(double), cudaMemcpyDeviceToDevice);
    }
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("\n Part 3_many blocks, time: \t%lf \n", cpu_time_used);
    cudaFree(u_dev);
    cudaFree(u1_dev);
    cudaFree(u2_dev);
    free(u);
    free(u1);
    free(u2);
}