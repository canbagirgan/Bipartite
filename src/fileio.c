#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>
#include "../includes/fileio.h"
#include "../includes/typedef.h"

//Local function for generating transpose of sparse matrix




// Oluşturulan bin dosyasına göre yeni paralel read fonksiyonu yazılacak
// l2g_map'in generate edildiği kısımlar çıkaralıcak
// l2g_map dosyadan okunulacak
// küçük m dosyadan okunacak veya hesaplancak
SparseMat* readSparseMat(char* fName, int partScheme, char* inPartFile) {
    if (partScheme == STORE_BY_COLUMNS) {
        printf("STORE_BY_COLUMNS not implemented.");
        exit (EXIT_FAILURE);
    } else {
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

        int64_t sloc;

        SparseMat* A = (SparseMat*) malloc(sizeof(SparseMat));

        FILE *fpmat = fopen(fName, "rb");
        fread(&(A->gm), sizeof(int), 1, fpmat);
        fread(&(A->gn), sizeof(int), 1, fpmat);
        fseek(fpmat, 2*sizeof(int)+(world_rank*sizeof(int64_t)), SEEK_SET);
        
        fread(&sloc, sizeof(int64_t), 1, fpmat);

        fseek(fpmat, sloc, SEEK_SET);
        fread(&(A->m), sizeof(int), 1, fpmat);
        fread(&(A->nnz), sizeof(int), 1, fpmat);
        fread(&(A->littleM), sizeof(int), 1, fpmat);

        sparseMatInit(A); // To initialize A we need gm gn m and nnz
        if (A->init != 0)
        {
            printf("Problem occured during initilization of Sparse Matrix A!");
        }
        fread(A->ia, sizeof(int), A->m+1, fpmat);
        fread(A->ja, sizeof(int), A->nnz, fpmat);
        fread(A->val, sizeof(double), A->nnz, fpmat);
        fread(A->l2gMap, sizeof(int), A->m, fpmat);

        A->store = STORE_BY_ROWS;

        A->inPart  = malloc(sizeof(*(A->inPart)) * A->gn);
        
        
        FILE *pf = fopen(inPartFile, "r");
        for (int i = 0; i < A->gn; ++i)
            fscanf(pf, "%d", &(A->inPart[i]));
        fclose(pf);

        // local n hesaplama
        int *tmp = malloc(sizeof(*tmp) * A->gn);
        memset(tmp, 0, sizeof(*tmp)*A->gn);
        A->n = 0;
        for (int i = 0; i < A->m; ++i) {
            for (int j = A->ia[i]; j < A->ia[i+1]; ++j)
                ++(tmp[A->ja[j]]);
        }

        for (int j = 0; j < A->gn; ++j) {
            if (world_rank == A->inPart[j])
                ++(tmp[j]);
        }

        for (int j = 0; j < A->gn; ++j) {
            if (tmp[j])
                ++(A->n);
        }
        // buraya kadar local n hesabı

        
        free(tmp);

        fclose(fpmat);

        return A;
    }
}

ParMatrix* readDenseMat(char* fName, SparseMat* A) {

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    char line[MAXCHAR];
    char *ptr;
    ParMatrix* X = (ParMatrix*) malloc(sizeof(ParMatrix));

    FILE* file = fopen(fName, "r");
    fgets(line, MAXCHAR, file);
    int gm = atoi(line);
    fgets(line, MAXCHAR, file);
    int feat_size = atoi(line);
    X->gm = gm;
    X->gn = feat_size;
    
    X->mat = matrix_create(A->littleM, feat_size);

    X->store = STORE_BY_ROWS;
    X->inPart = A->inPart;
    X->l2gMap = A->l2gMap;
    int r_ctr = 0;
    for (int i = 0;i<X->gm;i++) {

        if(X->inPart[i] == world_rank) {
            int c_ctr = 0;
            char *tok;
            fgets(line, MAXCHAR, file);
            for (tok = strtok(line, ",");tok && *tok;tok = strtok(NULL, ",")){
                X->mat->entries[r_ctr][c_ctr++] = strtod(tok, &ptr);
            }
            //printf("%d\n", c_ctr);
            r_ctr++;
        } else {
            fgets(line, MAXCHAR, file);
        }

    }
    fclose(file);
    return X;
}
