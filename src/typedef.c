#include "../includes/typedef.h"
#include <stdlib.h>
#include <string.h>

void sparseMatInit(SparseMat* A) {
    A->ia = (int *) malloc(sizeof(int) * (A->m + 1));
    A->ja = (int *) malloc(sizeof(int) * A->nnz);
    A->ja_mapped = (int *) malloc(sizeof(int) * A->nnz);
    A->val = (double *) malloc(sizeof(double) * A->nnz);
    //l2g_map allocate edilecek
    A->l2gMap = (int *) malloc(sizeof(int) * A->m);
    A->init = 0;
}

void sparseMatFree(SparseMat* A) {
    free(A->ia);
    free(A->ja);
    free(A->val);
    free(A->inPart);
    free(A->l2gMap);
    free(A);
    A = NULL;
}

/*
void generate_parCSR(SparseMat* A, int* recv_map, int world_size,int world_rank) {
    int i, j;
    
    int* procMap = (int *) malloc(sizeof(int) * (world_size + 1));
    A->proc_map = (int *) malloc(sizeof(int) * (world_size + 1));
    int* temp_proc = (int *) malloc(sizeof(int) * (world_size));
    memset(procMap, 0, sizeof(int) * (world_size + 1));
    memset(A->proc_map, 0, sizeof(int) * (world_size + 1));
    memset(temp_proc, -1, sizeof(int) * (world_size));
    
    int v_j, part;
    for (i=0;i < A->m; i++) {
        for (j=A->ia[i];j < A->ia[i+1]; j++) {
            v_j = A->ja[j];
            part = A->inPart[v_j];
            procMap[part+1] += 1;
            if (temp_proc[part] != i) {
                temp_proc[part] = i;
                A->proc_map[part+1] += 1;
            }
        }
    }
    
    
    for (i=0;i<world_size;i++) {
        procMap[i + 1] += procMap[i];
        A->proc_map[i + 1] += A->proc_map[i];
    }
    
    A->ic = (csrPtr *) malloc(sizeof(csrPtr) * (A->proc_map[world_size] + 1));
    A->jc = (int *) malloc(sizeof(int) * A->nnz);
    A->jc_mapped = (int *) malloc(sizeof(int) * A->nnz);
    A->val_c = (double *) malloc(sizeof(double) * A->nnz);
    memset(temp_proc, -1, sizeof(int) * (world_size));


    for (i=0;i < A->m; i++) {
        for (j=A->ia[i];j < A->ia[i+1]; j++) {      
            v_j = A->ja[j];
            part = A->inPart[v_j];       
            if (temp_proc[part] != i) {
                A->ic[A->proc_map[part]].indptr = procMap[part];
                A->ic[A->proc_map[part]].v_id = i;
                A->proc_map[part] += 1;     
                temp_proc[part] = i;
            }
            A->jc[procMap[part]] = v_j;
            A->jc_mapped[procMap[part]] = recv_map[v_j];
            A->val_c[procMap[part]] = A->val[j];
            procMap[part] += 1;
            
        }
    }
    A->ic[A->proc_map[world_size]].indptr = procMap[world_size];
    A->ic[A->proc_map[world_size]].v_id = -1;
    
    for (i=world_size;i>0;i--) {
        A->proc_map[i] = A->proc_map[i-1];
    }
    A->proc_map[0] = 0;
}
*/
Matrix* matrix_create(int row, int col) {
    Matrix *matrix = (Matrix *) malloc(sizeof(Matrix));
    matrix->m = row;
    matrix->n = col;
    int total = row * col;

    double *data = (double *) calloc(total, sizeof(double));
    matrix->entries = (double **) malloc(row * sizeof(double*));

    for (int i = 0; i < row; i++) {
        matrix->entries[i] = &(data[col*i]);
        //memset(matrix->entries[i], 0, col * sizeof(double));
    }
    return matrix;
}

void matrix_free(Matrix *m) {
    free(m->entries[0]);
    free(m->entries);
    free(m);
    m = NULL;
}

ParMatrix* init_ParMatrix(SparseMat* A, int n) {
    ParMatrix* X = (ParMatrix*) malloc(sizeof(ParMatrix));
    X->gm = A->gm;
    X->gn = n;
    X->mat = matrix_create(A->m, n);
    X->store = STORE_BY_ROWS;
    X->inPart = A->inPart;
    X->l2gMap = A->l2gMap;

    return X;
}

void parMatrixFree(ParMatrix* X) {
    matrix_free(X->mat);
    free(X);
    X = NULL;
}

sendTable* sendTableCreate(int p_count, int myId, int n) {
    /*
    p_count => world size
    myId => processor id
    n => A->m
    */
    sendTable* table = (sendTable*) malloc(sizeof(sendTable));
    table->p_count = p_count;
    table->myId = myId;
    table->n = n;



    table->send_count = (int *) malloc(p_count * sizeof(int));
    memset(table->send_count, 0, sizeof(int) * p_count);



    table->table = (int **) malloc(p_count * sizeof(int *));
    for (int i=0;i < p_count; i++) {
        table->table[i] = (int *) malloc(n * sizeof(int));

        memset(table->table[i], 0, sizeof(int) * n);
    }

    return table;
}

void sendTableFree(sendTable* table) {
    for (int i=0;i < table->p_count; i++) {
        free(table->table[i]);
    }
    free(table->table);
    free(table->send_count);
    free(table);
    table = NULL;
}

recvTable* recvTableCreate(int p_count, int myId, int n) {
    /*
    p_count => world size
    myId => processor id
    n => A->m
    */
    recvTable* table = (recvTable*) malloc(sizeof(recvTable));
    table->p_count = p_count;
    table->myId = myId;
    table->n = n;

    table->recv_count = (int *) malloc(p_count * sizeof(int));
    if(table->recv_count != NULL) {
        memset(table->recv_count, 0, sizeof(*table->recv_count) * p_count);
    }

    table->table = (int **) malloc(p_count * sizeof(int *));
    for (int i=0;i < p_count; i++) {
        table->table[i] = (int *) malloc(n * sizeof(int ));
        if(table->table[i] != NULL) {
            memset(table->table[i], 0, sizeof(*table->table[i]) * n);
        }
    }

    return table;
}

void recvTableFree(recvTable* table) {
    for (int i=0;i < table->p_count; i++) {
        free(table->table[i]);
    }
    free(table->table);
    free(table->recv_count);
    free(table);
    table = NULL;
}
/*
No longer needed
sendBuffer* sendBufferCreate(int send_count, int feature_size, int p_id) {
    sendBuffer* buffer = (sendBuffer *) malloc(sizeof(sendBuffer));
    buffer->send_count = send_count;


    buffer->feature_size = feature_size;
    buffer->p_id = p_id;
    buffer->vertices = (int *) malloc(sizeof(int) * send_count);
    buffer->vertices_local = (int *) malloc(sizeof(int) * send_count);
    //buffer->data = (double **) malloc(sizeof(double *) * send_count);
    //double *data = (double *) malloc(send_count * feature_size * sizeof(double));

    //for (int j = 0;j < send_count; j++) {
        //buffer->data[j] = &(data[feature_size * j]);
    //}
    return buffer;
}
*/
void initSendBufferSpace(sendBuffer* buffer) {
    buffer->data = (double **) malloc(sizeof(double *) * buffer->send_count);
    double *data = (double *) malloc(buffer->send_count * buffer->feature_size * sizeof(double));

    for (int j = 0;j < buffer->send_count; j++) {
        buffer->data[j] = &(data[buffer->feature_size * j]);
    }
}

void sendBufferSpaceFree(sendBuffer* buffer) {
    if (buffer->send_count == 0) {
    	buffer->data[0] = NULL;
    	buffer->data = NULL;
    	return;
    }
    free(buffer->data[0]);
    free(buffer->data);
}

void sendBufferFree(sendBuffer* buffer) {
    free(buffer->data[0]);
    free(buffer->data);
    free(buffer->vertices);
    free(buffer);
    buffer = NULL;
}
/*
void sendBufferListFree(sendBuffer** bufferList, int world_size, int world_rank) {
    for(int i=0;i< world_size;i++) {
        if (i != world_rank) {
            sendBufferFree(bufferList[i]);
        }
    }
    bufferList = NULL;
}
*/
/*
No longer needed
recvBuffer* recvBufferCreate(int recv_count, int feature_size, int p_id) {
    recvBuffer* buffer = (recvBuffer *) malloc(sizeof(recvBuffer));
    buffer->recv_count = recv_count;
    buffer->feature_size = feature_size;
    buffer->p_id = p_id;
    buffer->vertices = (int *) malloc(sizeof(int) * recv_count);
    
    buffer->data = (double **) malloc(sizeof(double *) * recv_count);
    double *data = (double *) malloc(recv_count * feature_size * sizeof(double));

    for (int j = 0;j < recv_count; j++) {
        buffer->data[j] = &(data[feature_size * j]);
    }
    
    return buffer;
}
*/
void initRecvBufferSpace(recvBuffer* buffer) {
    buffer->data = (double **) malloc(sizeof(double *) * buffer->recv_count);
    double *data = (double *) malloc(buffer->recv_count * buffer->feature_size * sizeof(double));

    for (int j = 0;j < buffer->recv_count; j++) {
        buffer->data[j] = &(data[buffer->feature_size * j]);
    }
}

void recvBufferSpaceFree(recvBuffer* buffer) {
	if (buffer->recv_count == 0) {
    	buffer->data[0] = NULL;
    	buffer->data = NULL;
    	return;
    }
    free(buffer->data[0]);
    free(buffer->data);
}

void recvBufferFree(recvBuffer* buffer) {
    free(buffer->data[0]);
    free(buffer->data);
    free(buffer->vertices);
    free(buffer);
    buffer = NULL;
}
/*
void recvBufferListFree(recvBuffer** bufferList, int world_size, int world_rank) {
    for(int i=0;i< world_size;i++) {
        if (i != world_rank) {
            recvBufferFree(bufferList[i]);
        }
    }
    bufferList = NULL;
}
*/
void initBufferSpace(Buffer* buffer) {
    buffer->data = (double **) malloc(sizeof(double *) * buffer->count);
    double *data = (double *) malloc(buffer->count * buffer->feature_size * sizeof(double));
    buffer->reqs = (MPI_Request*) malloc(buffer->count * sizeof(MPI_Request));
    
    for (int j = 0;j < buffer->count; j++) {
        buffer->data[j] = &(data[buffer->feature_size * j]);
    }
}

void bufferSpaceFree(Buffer* buffer) {
	if (buffer->count == 0) {
    	buffer->data[0] = NULL;
    	buffer->data = NULL;
    	return;
    }
    free(buffer->data[0]);
    free(buffer->data);
}

void bufferFree(Buffer* buffer) {
    if (buffer->count == 0) {
    	buffer->data[0] = NULL;
    	buffer->data = NULL;
        buffer->vertices = NULL;
        buffer->vertices_local = NULL;
        buffer->reqs = NULL;
        buffer->ll_vertices = NULL;
        buffer->pid_map = NULL;
        buffer->list = NULL;
        buffer = NULL;
    	return;
    }
    free(buffer->data[0]);
    buffer->data[0] = NULL;
    free(buffer->data);
    buffer->data = NULL;
    free(buffer->vertices);
    buffer->vertices = NULL;
    //free(buffer->vertices_local);
    //buffer->vertices_local = NULL;
    free(buffer->reqs);
    buffer->reqs = NULL;
    free(buffer->ll_vertices);
    buffer->ll_vertices = NULL;
    free(buffer->pid_map);
    buffer->pid_map = NULL;
    //free(buffer->list);
    //buffer->list = NULL;
    free(buffer);
    buffer = NULL;
}