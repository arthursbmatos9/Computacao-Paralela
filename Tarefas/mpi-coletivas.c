#include <time.h>
#include <stdio.h>
#include <mpi.h>
#define N 10
#define MAX 4
#define NUMBER 3
void main(int argc, char *argv[])
{
    int p, rank, maior_parcial, maior_final, numProcs;
    int buffer[N];
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    if (rank == 0)
    {
        // preencher o buffer com N valores inteiros aleatórios
        for (int i = 0; i < N; i++) {
            buffer[i] = rand() % 100;
        }
    }

    // distribuir o vetor para todos os outros processos
    MPI_Bcast(buffer, N, MPI_INT, 0, MPI_COMM_WORLD);
    
    // processar o maior dos valores dentro do seu intervalo
    int elementos_processo = N / numProcs;
    int inicio = rank * elementos_processo;
    int fim = (rank == numProcs - 1) ? N : (rank + 1) * elementos_processo;
    
    maior_parcial = buffer[inicio];
    for (int i = inicio + 1; i < fim; i++) {
        if (buffer[i] > maior_parcial) {
            maior_parcial = buffer[i];
        }
    }

    // reduzir os maiores no maior, enviando o resultado para o processo com rank = 0
    MPI_Reduce(&maior_parcial, &maior_final, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0)
    {
        // imprimir maior
        printf("\nMaior valor: %d\n", maior_final);
    }
    MPI_Finalize();
}
