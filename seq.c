#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define n 0.0002
#define p 0.5
#define G 0.75

#define SIZE 512
#define NUMBER_OF_ITERATIONS 10

int idx(int i, int j){
    return (SIZE * i + j);
}

void printMatrix(double* u){
        printf("\n");
        for(int i = 0; i < SIZE * SIZE; i++){
            printf("%.3lf", u[i]);
            printf("\t");
            if((i+1) %  4 == 0 && i > 0){
                printf("\n");
            }
        }
}

int main(){


    double* u1 = malloc(sizeof(double) * SIZE * SIZE );
    double* u2 = malloc(sizeof(double) * SIZE * SIZE );
    double* u = malloc(sizeof(double) * SIZE * SIZE );


    //initialize to 0
    for(int i = 0; i < SIZE * SIZE; i++){
        u1[i] = 0;
        u2[i] = 0;
        u[i] = 0;
    }

    //hit that drummmm
    u1[idx(SIZE/2, SIZE/2)] = 1;
    
    //printMatrix(u1);
    
    clock_t start, end;
    double cpu_time_used;
     
    start = clock();

    for(int a=0; a < NUMBER_OF_ITERATIONS; a++)
    {    
        for(int i = 1; i <SIZE-1; i++){
            for(int j = 1; j <SIZE-1 ; j++){

            u[idx(i, j)]=  p * 
                                    (u1[idx(i-1,j)] + u1[idx(i+1,j)] 
                                    +u1[idx(i,j-1)] + u1[idx(i,j+1)] 
                                    - 4 * u1[idx(i, j)])  
                                + 2 * u1[idx(i, j)] - (1-n) * u2[idx(i, j)];
            }
        }

        for(int i = 1; i < SIZE-1; i++){
            u[idx(0, i)]       = G * u[idx(1, i)];
            u[idx(SIZE-1, i)]  = G * u[idx(SIZE-2, i)];
            u[idx(i,0)]        = G * u[idx(i, 1)];
            u[idx(i, SIZE-1)]  = G * u[idx(i, SIZE-2)];
        }

        u[idx(0,0)]            = G * u[idx(1,0)];
        u[idx(SIZE-1,0)]       = G * u[idx(SIZE-2, 0)];
        u[idx(0,SIZE-1)]       = G * u[idx(0, SIZE-2)];
        u[idx(SIZE-1, SIZE-1)] = G * u[idx(SIZE-1, SIZE-2)];

        
        
        //printMatrix(u);
        printf("\n\n%lf", u[idx(SIZE/2, SIZE/2)]);

        
        for(int i=0; i < SIZE * SIZE; i++){
            //u[i] = u[i];
            u2[i] = u1[i];
            u1[i] = u[i];
        }

    }


     end = clock();
     cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("\nExecution time: \t%lf \n", cpu_time_used);
    free(u1);
    free(u2);
    free(u);
}