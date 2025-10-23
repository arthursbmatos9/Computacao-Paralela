#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

/* DETALHES DA ARQUITETURA

Processador: 11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz (2.80 GHz)
Tipo de sistema:	Sistema operacional de 64 bits, processador baseado em x64
SO: Ubuntu 22.04.2 LTS (WSL)
Cores: 4
Threads: 8
RAM:	16,0 GB


Tempo sequencial: 2.392s
Tempo paralelo schedule(static, 5): 1.128s   ---> speedup = 2.12x
Tempo paralelo schedule(dynamic, 5): 1.088s  ---> speedup = 2.19x
Tempo paralelo schedule(guided, 25): 1.833s  ---> speedup = 1.30x
*/
int sieveOfEratosthenes(int n)
{
   int primes = 0; 
   bool *prime = (bool*) malloc((n+1)*sizeof(bool));
   int sqrt_n = sqrt(n);

   memset(prime, true,(n+1)*sizeof(bool));

   #pragma omp parallel for reduction(+:primes) schedule(guided, 25)
   for (int p=2; p <= sqrt_n; p++)
   {
       // If prime[p] is not changed, then it is a prime
       if (prime[p] == true)
       {
           // Update all multiples of p
           for (int i=p*2; i<=n; i += p)
           prime[i] = false;
        }
    }

    // count prime numbers
    for (int p=2; p<=n; p++)
       if (prime[p])
         primes++;

    return(primes);
}

int main()
{
   int n = 100000000;
   printf("%d\n",sieveOfEratosthenes(n));
   return 0;
} 