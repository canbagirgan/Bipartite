#ifndef TYPEDEF_H_INCLUDED
#define TYPEDEF_H_INCLUDED
#include <mpi.h>

#pragma once

#define STORE_BY_COLUMNS 0
#define STORE_BY_ROWS    1

#define AGG_COMM 0

typedef struct {
    double commTimeF;
    double commTimeB;
    double compTimeF;
    double compTimeB;
    double compTime;
    double commTime;
    double other;
    double totalTime;
} Timer;

typedef struct {
    int sendVol;
    int recvVol;
    int sendRecvVol;
    int multiplyAdd;
} Stats;

typedef struct {
    int indptr;
    int v_id;
} csrPtr;

typedef struct {
    int *ia;    // rows of A in csr format
    int *ja;    // cols of A in csr format
    int *ja_mapped; // cols of A in csr format locally mapped
    double *val; // values of A in csr format
    
    int n; // owned and borrowed nodes number of partition
    int m; 
    int nnz;
    int littleM; // owned nodes number of partition
    int gn, gm;
    int store;
    int *l2gMap;
    int *inPart;

    int init;
} SparseMat;

typedef struct {
    double** entries;
    int m;
    int n;
} Matrix;

typedef struct {
    Matrix* mat;
    int gm;
    int gn;
    int store;
    int *l2gMap;
    int *inPart;
} ParMatrix;

typedef struct {
    int p_count;
    int myId;
    int n;
    int *send_count;
    int **table;
} sendTable;

typedef struct {
    int p_count;
    int myId;
    int n;
    int *recv_count;
    int **table;
} recvTable;

typedef struct {
    int send_count; //m - littleM
    int *pid_map;
    int feature_size;
    int *vertices;
    int *vertices_local;
    double **data;
} sendBuffer;

typedef struct {
    int recv_count;
    int *pid_map;
    int feature_size;
    int *vertices;
    int *vertices_local;
    double **data;
} recvBuffer;

typedef struct {
    int count;
    int *pid_map;
    int feature_size;
    int *vertices;
    int *vertices_local;
    int *list;
    int ll_count;
    int *ll_vertices;
    MPI_Request *reqs;
    double **data;
} Buffer;

void sparseMatInit(SparseMat* A);
void sparseMatFree(SparseMat* A);
void csrToCsc(SparseMat* A);
void generate_parCSR(SparseMat* A, int* recv_map, int world_size,int world_rank);

Matrix* matrix_create(int row, int col);
void matrix_free(Matrix *m);
ParMatrix* init_ParMatrix(SparseMat* A, int n);
void parMatrixFree(ParMatrix* X);

sendTable* sendTableCreate(int p_count, int myId, int n); //n equals A->m
void sendTableFree(sendTable* table);

recvTable* recvTableCreate(int p_count, int myId, int n); //n equals A->gn
void recvTableFree(recvTable* table);

//sendBuffer* sendBufferCreate(int send_count, int feature_size, int p_id);
void sendBufferFree(sendBuffer* buffer);
void initSendBufferSpace(sendBuffer* buffer);
void sendBufferSpaceFree(sendBuffer* buffer);
//void sendBufferListFree(sendBuffer** bufferList, int world_size, int world_rank);

//recvBuffer* recvBufferCreate(int send_count, int feature_size, int p_id);
void initRecvBufferSpace(recvBuffer* buffer);
void recvBufferSpaceFree(recvBuffer* buffer);
void recvBufferFree(recvBuffer* buffer);
//void recvBufferListFree(recvBuffer** bufferList, int world_size, int world_rank);

void initBufferSpace(Buffer* buffer);
void bufferSpaceFree(Buffer* buffer);
void bufferFree(Buffer* buffer);
#endif // TYPEDEF_H_INCLUDED
