#include <stdio.h>

#include "../includes/utils.h"

void printSparseMatrix(SparseMat* matrix)
{
    printf("Local m: %d", matrix->m);
    printf("Local nnz: %d", matrix->nnz);
    printf("Little m: %d", matrix->littleM);
}

