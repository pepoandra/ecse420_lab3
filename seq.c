#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// #define n 0.0002
// #define p 0.5
// #define G 0.75

#define SIZE 4


int idx(int i, int j){
    return (SIZE * i + j);
}

int main(){

    double n = 0.0002;
    double p = 0.5;
    double G = 0.75;

    double* u = malloc(sizeof(double) * SIZE * SIZE );
    double* u1 = malloc(sizeof(double) * SIZE * SIZE );
    double* u2 = malloc(sizeof(double) * SIZE * SIZE );
    double* buffer = malloc(sizeof(double) * SIZE * SIZE );


    //initialize to 0
    for(int i = 0; i < SIZE * SIZE; i++){
        u[i] = 0;
        u1[i] = 0;
        u2[i] = 0;
        buffer[i] = 0;
    }

    //hit that drummmm
    //arr[SIZE * SIZE/2 + SIZE/2] = 1;
    u1[idx(SIZE/2, SIZE/2)] = 1;
    
    for(int i = 0; i < SIZE * SIZE; i++){
        printf("%lf", u1[i]);
        printf("\t");
        if((i+1) %  4 == 0 && i > 0){
            printf("\n");
        }
    }
    

     clock_t start, end;
     double cpu_time_used;
     
     start = clock();
     

    for(int a=0; a < 3; a++)
    {    
        for(int i = 1; i <SIZE-1; i++){
            for(int j = 1; j <SIZE-1 ; j++){

            buffer[idx(i, j)]=  p * 
                                    (u1[idx(i-1,j)] + u1[idx(i+1,j)] 
                                    +u1[idx(i,j-1)] + u1[idx(i,j+1)] 
                                    - 4 * u1[idx(i, j)])  
                                + 2 * u1[idx(i, j)] - (1-n) * u2[idx(i, j)];
            }
        }

        for(int i = 1; i < SIZE-1; i++){
            buffer[idx(0, i)]       = G * buffer[idx(1, i)];
            buffer[idx(SIZE-1, i)]  = G * buffer[idx(SIZE-2, i)];
            buffer[idx(i,0)]        = G * buffer[idx(i, 1)];
            buffer[idx(i, SIZE-1)]  = G * buffer[idx(i, SIZE-2)];
        }

        buffer[idx(0,0)]            = G * buffer[idx(1,0)];
        buffer[idx(SIZE-1,0)]       = G * buffer[idx(SIZE-2, 0)];
        buffer[idx(0,SIZE-1)]       = G * buffer[idx(0, SIZE-2)];
        buffer[idx(SIZE-1, SIZE-1)] = G * buffer[idx(SIZE-1, SIZE-2)];

        printf("\n");
        for(int i = 0; i < SIZE * SIZE; i++){
            printf("%.10lf", buffer[i]);
            printf("\t");
            if((i+1) %  4 == 0 && i > 0){
                printf("\n");
            }
        }

        for(int i=0; i < SIZE * SIZE; i++){
            u[i] = buffer[i];
            u2[i] = u1[i];
        }

        for(int i=0; i < SIZE * SIZE; i++){
            u1[i] = u[i];
        }
    }


     end = clock();
     cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("\nExecution time: \t%lf \n", cpu_time_used);
    free(u);
    free(u1);
    free(u2);
    free(buffer);
}