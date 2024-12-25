#include "../includes/basic.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <mpi.h>

Buffer* initSendBuffer(SparseMat* A, int *l2gMap, int feature_size) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    Buffer* buffer = (Buffer *) malloc(sizeof(Buffer));
    buffer->feature_size = feature_size;
    
    buffer->count = A->m - A->littleM;

    //printf("M: %d, Little M: %d\n", A->m, A->littleM);
    //printf("Send count: %d\n", buffer->send_count);
    // littleM'inci vertexten m'inci vertexe kadar olanları 
    //  bufferın verticesine ekle(global indekslerini)
    buffer->vertices = (int *) malloc(sizeof(int) * buffer->count );

    int ind = 0;    
    for (int i=A->littleM; i<A->m; i++) {
        buffer->vertices[ind++] = l2gMap[i];
    }
    
    // ???buffer->vertices_local = (int *) malloc(sizeof(int) * buffer->send_count );
    // her processorun send edeceği vektörün başlangıç indisini tutan bir map 
    // processorlere göre sıralı oldukları için ia'larından littleM+i şeklinde iterate ederek çekebiliriz


    buffer->pid_map = (int *) malloc(sizeof(int) * (world_size + 1) );
    buffer->pid_map[0] = 0;
    //memset(buffer->pid_map, 0, world_size + 1);
    for (int i=0; i<world_size+1; i++){
        buffer->pid_map[i] = 0;
    }

    int counter = 0;
    //printf("pid: %d\n", A->inPart[buffer->vertices[0]]);
    //int processor_id = A->inPart[buffer->vertices[0]];

    // fill the pid_map with send counts of each proc
    for (int i=0; i<buffer->count; i++) {
        counter = A->inPart[buffer->vertices[i]];
        buffer->pid_map[counter+1]++;
    }

    //prefix sum
    for(int i=1; i<world_size+1; i++) {
        buffer->pid_map[i] += buffer->pid_map[i-1];
    }

    return buffer;
}

Buffer* initRecvBuffer(SparseMat* A, int feature_size, Buffer* sendBuff) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    Buffer* buffer = (Buffer*) malloc(sizeof(Buffer));
    buffer->feature_size = feature_size;
    buffer->pid_map = (int*) malloc(sizeof(int) * (world_size+1));

    memset(buffer->pid_map, 0,  sizeof(int) * (world_size+1));


    buffer->count = 0;

    //MPI_Request sreq[world_size-1];
    MPI_Request rreq[world_size-1];

    MPI_Barrier(MPI_COMM_WORLD);
    int c_ctr=0;
    for (int i=0; i<world_size; i++) {
        if (i != world_rank) {
            MPI_Irecv(&buffer->pid_map[i+1], 1, MPI_INT, i, 0, MPI_COMM_WORLD, &rreq[c_ctr]);
            //MPI_Recv(&buffer->pid_map[i+1], 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int send_count = sendBuff->pid_map[i+1] - sendBuff->pid_map[i];
            //MPI_Isend(&send_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &sreq[c_ctr]);
            MPI_Send(&send_count, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
            c_ctr++;
        }

    }
    //MPI_Barrier(MPI_COMM_WORLD);
    MPI_Waitall(c_ctr, rreq, MPI_STATUS_IGNORE);
    //MPI_Waitall(c_ctr, sreq, MPI_STATUS_IGNORE);


    for(int i=1; i<world_size+1; i++) {
        buffer->pid_map[i] += buffer->pid_map[i-1];
    }
    buffer->count = buffer->pid_map[world_size];

    /* For debug
    printf("Recv count: %d\n", buffer->recv_count);
    for(int i=0; i<world_size+1; i++) {
        printf("%d ", buffer->pid_map[i]);
    }
    printf("\n");
    */

    buffer->vertices = (int *) malloc(sizeof(int) * buffer->count);
    memset(buffer->vertices, 0, buffer->count * sizeof(int));

    c_ctr = 0;
    for (int i=0; i<world_size; i++) {
        if (i != world_rank) {
            int recv_count = buffer->pid_map[i+1] - buffer->pid_map[i];
            MPI_Irecv(&(buffer->vertices[buffer->pid_map[i]]), recv_count, MPI_INT, i, 0, MPI_COMM_WORLD, &rreq[c_ctr]);
            //MPI_Recv(&(buffer->vertices[buffer->pid_map[i]]), recv_count, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int send_count = sendBuff->pid_map[i+1] - sendBuff->pid_map[i];
            //printf("%d ", send_count);
            //MPI_Isend(&(sendBuff->vertices[sendBuff->pid_map[i]]), send_count, MPI_INT, i, 0, MPI_COMM_WORLD, &sreq[c_ctr]);
            MPI_Send(&(sendBuff->vertices[sendBuff->pid_map[i]]), send_count, MPI_INT, i, 0, MPI_COMM_WORLD);
            c_ctr++;
        }
    }
    MPI_Waitall(c_ctr, rreq, MPI_STATUS_IGNORE);
    //MPI_Waitall(c_ctr, sreq, MPI_STATUS_IGNORE);

    
    /**
    if (world_rank == 17) {
        printf("\nVertices: ");
        for (int i=0; i<buffer->count; i++) {
            printf("%d ", buffer->vertices[i]);
        }
        printf("\n");
        printf("\nPidMap: ");
        for (int i=0; i<world_size+1; i++) {
            printf("%d ", buffer->pid_map[i]);
        }
        printf("\n");
    }
    */
    return buffer;
}