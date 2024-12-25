#include "../includes/basic.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

sendTable* initSendTable(SparseMat* A) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    sendTable* table = sendTableCreate(world_size, world_rank, A->m);

    for (int i = 0; i < A->m; i++) {
        for (int j = A->ia[i]; j < A->ia[i+1]; j++) {
            int u_id = A->inPart[A->ja[j]];
            if (u_id != world_rank) {
                if (table->table[u_id][i] == 0) {
                    table->table[u_id][i] = 1;
                    table->send_count[u_id]++;
                }
            }
        }
    }
    //printf("Send count in processor %d => %d\n", world_rank, table->send_count[1]);
    
    return table;
}

recvTable* initRecvTable(SparseMat* A_T) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    recvTable* table = recvTableCreate(world_size, world_rank, A_T->gn);

    for (int i = 0; i < A_T->m; i++) {
        for (int j = A_T->ia[i]; j < A_T->ia[i+1]; j++) {
            int u_id = A_T->inPart[A_T->ja[j]];
            if (u_id != world_rank) {
                if (table->table[u_id][A_T->ja[j]] == 0) {
                    table->table[u_id][A_T->ja[j]] = 1;
                    table->recv_count[u_id]++;
                }
            }
        }
    }
    //printf("Send count in processor %d => %d\n", world_rank, table->send_count[1]);

    return table;
}

sendBuffer* initSendBuffer(SparseMat* A, int *l2gMap, int feature_size) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    sendBuffer* buffer = (sendBuffer *) malloc(sizeof(sendBuffer));
    buffer->feature_size = feature_size;
    
    buffer->send_count = A->m - A->littleM;

    //printf("M: %d, Little M: %d\n", A->m, A->littleM);
    //printf("Send count: %d\n", buffer->send_count);
    // littleM'inci vertexten m'inci vertexe kadar olanları 
    //  bufferın verticesine ekle(global indekslerini)
    buffer->vertices = (int *) malloc(sizeof(int) * buffer->send_count );

    int ind = 0;    
    for (int i=A->littleM; i<A->m; i++) {
        buffer->vertices[ind++] = l2gMap[i];
    }
    
    //???buffer->vertices_local = (int *) malloc(sizeof(int) * buffer->send_count );
    // her processorun send edeceği vektörün başlangıç indisini tutan bir map 
    // processorlere göre sıralı oldukları için ia'larından littleM+i şeklinde iterate ederek çekebiliriz


    buffer->pid_map = (int *) malloc(sizeof(int) * (world_size + 1) );
    buffer->pid_map[0] = 0;
    //memset(buffer->pid_map, 0, world_size + 1);
    for (int i=0; i<world_size+1; i++){
        buffer->pid_map[i] = 0;
    }

    int counter = 0;
    int processor_id = A->inPart[buffer->vertices[0]];
    // fill the pid_map with send counts of each proc
    for (int i=0; i<buffer->send_count; i++) {
        counter = A->inPart[buffer->vertices[i]];
        buffer->pid_map[counter+1]++;
    }

    //prefix sum
    for(int i=1; i<world_size+1; i++) {
        buffer->pid_map[i] += buffer->pid_map[i-1];
    }

    return buffer;
}

recvBuffer* initRecvBuffer(SparseMat* A, int feature_size, sendBuffer* sendBuff) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    recvBuffer* buffer = (recvBuffer*) malloc(sizeof(recvBuffer));
    buffer->feature_size = feature_size;
    buffer->pid_map = (int*) malloc(sizeof(int) * (world_size+1));

    memset(buffer->pid_map, 0,  sizeof(int) * (world_size+1));


    buffer->recv_count = 0;

    MPI_Request sreq[world_size-1];
    MPI_Request rreq[world_size-1];

    MPI_Barrier(MPI_COMM_WORLD);
    int c_ctr=0;
    for (int i=0; i<world_size; i++) {
        if (i != world_rank) {
            MPI_Irecv(&buffer->pid_map[i+1], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &rreq[c_ctr]);
            int send_count = sendBuff->pid_map[i+1] - sendBuff->pid_map[i];
            MPI_Isend(&send_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &sreq[c_ctr]);
            c_ctr++;
        }

    }
    //MPI_Barrier(MPI_COMM_WORLD);
    MPI_Waitall(world_size-1, rreq, MPI_STATUS_IGNORE);
    MPI_Waitall(world_size-1, sreq, MPI_STATUS_IGNORE);


    for(int i=1; i<world_size+1; i++) {
        buffer->pid_map[i] += buffer->pid_map[i-1];
    }
    buffer->recv_count = buffer->pid_map[world_size];

    /* For debug
    printf("Recv count: %d\n", buffer->recv_count);
    for(int i=0; i<world_size+1; i++) {
        printf("%d ", buffer->pid_map[i]);
    }
    printf("\n");
    */

    buffer->vertices = (int *) malloc(sizeof(int) * buffer->recv_count);
    memset(buffer->vertices, 0, buffer->recv_count * sizeof(int));

    c_ctr = 0;
    for (int i=0; i<world_size; i++) {
        if (i != world_rank) {
            int recv_count = buffer->pid_map[i+1] - buffer->pid_map[i];
            MPI_Irecv(&(buffer->vertices[buffer->pid_map[i]]), recv_count, MPI_INT, i, 0, MPI_COMM_WORLD, &rreq[c_ctr]);

            int send_count = sendBuff->pid_map[i+1] - sendBuff->pid_map[i];
            //printf("%d ", send_count);
            MPI_Isend(&(sendBuff->vertices[sendBuff->pid_map[i]]), send_count, MPI_INT, i, 0, MPI_COMM_WORLD, &sreq[c_ctr]);
            c_ctr++;
        }
    }
    MPI_Waitall(world_size-1, rreq, MPI_STATUS_IGNORE);
    MPI_Waitall(world_size-1, sreq, MPI_STATUS_IGNORE);


    return buffer;
}
