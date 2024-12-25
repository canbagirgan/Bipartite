#include "../includes/sparseMat.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>



void aggregate(gcnLayer* layer, Matrix* X, Matrix* Y, int step) {

    int world_size, world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int i,j,k;
    sendBuffer* ph1_send_buffer;
    recvBuffer* ph1_recv_buffer;
    sendBuffer* ph2_send_buffer;
    recvBuffer* ph2_recv_buffer;

    SparseMat* A;

    int ph1_msg_send_count, ph1_msg_recv_count, ph2_msg_send_count, ph2_msg_recv_count;

    if (step == FORWARD) {
        A = layer->adjacency_T;
        ph1_send_buffer = layer->ph1_sendBuffer_forward;
        ph1_recv_buffer = layer->ph1_recvBuffer_forward;
        ph2_send_buffer = layer->ph2_sendBuffer_forward;
        ph2_recv_buffer = layer->ph2_recvBuffer_forward;
        ph1_msg_send_count = layer->ph1_msgSendCount_forward;
        ph1_msg_recv_count = layer->ph1_msgRecvCount_forward;
        ph2_msg_send_count = layer->ph2_msgSendCount_forward;
        ph2_msg_recv_count = layer->ph2_msgRecvCount_forward;
    }
    else if(step == BACKWARD) {
        A = layer->adjacency;
        ph1_send_buffer = layer->ph1_sendBuffer_backward;
        ph1_recv_buffer = layer->ph1_recvBuffer_backward;
        ph2_send_buffer = layer->ph2_sendBuffer_backward;
        ph2_recv_buffer = layer->ph2_recvBuffer_backward;
        ph1_msg_send_count = layer->ph1_msgSendCount_backward;
        ph1_msg_recv_count = layer->ph1_msgRecvCount_backward;
        ph2_msg_send_count = layer->ph2_msgSendCount_backward;
        ph2_msg_recv_count = layer->ph2_msgRecvCount_backward;
    }
    else {
        printf("ERROR");
        return;
    }
    
    int ind, ind_c;
    int range, base;

    MPI_Request* ph1_request_send = (MPI_Request*) malloc((ph1_msg_send_count) * sizeof(MPI_Request));
    MPI_Request* ph1_request_recv = (MPI_Request*) malloc((ph1_msg_recv_count) * sizeof(MPI_Request));

    MPI_Request* ph2_request_send = (MPI_Request*) malloc((ph2_msg_send_count) * sizeof(MPI_Request));
    MPI_Request* ph2_request_recv = (MPI_Request*) malloc((ph2_msg_recv_count) * sizeof(MPI_Request));

    // PHASE-1 COMMUNICATION (PRE-COMMUNICATION)

    ind_c = 0;
    initRecvBufferSpace(ph1_recv_buffer);
    for (i = 0;i < world_size; i++) {
        if (i != world_rank) {
            range = ph1_recv_buffer->pid_map[i+1] - ph1_recv_buffer->pid_map[i];
            base = ph1_recv_buffer->pid_map[i];
            if (range != 0) {
                MPI_Irecv(&(ph1_recv_buffer->data[base][0]),
                          range * ph1_recv_buffer->feature_size,
                          MPI_DOUBLE,
                          i,
                          AGG_COMM+i,
                          MPI_COMM_WORLD,
                          &(ph1_request_recv[ind_c]));
                ind_c++;
            }
        }
    }
    
    ind_c = 0;
    initSendBufferSpace(ph1_send_buffer);
    for (i=0;i < world_size; i++) {
        range = ph1_send_buffer->pid_map[i+1] - ph1_send_buffer->pid_map[i];
        base = ph1_send_buffer->pid_map[i];
        if (i != world_rank) {
            if (range != 0) {
                for (j = 0;j < range; j++) {
                    ind = ph1_send_buffer->vertices_local[base + j];
                    memcpy(ph1_send_buffer->data[base + j],  X->entries[ind] , sizeof(double) * ph1_send_buffer->feature_size);

                }

                MPI_Isend(&(ph1_send_buffer->data[base][0]),
                          range * ph1_send_buffer->feature_size,
                          MPI_DOUBLE,
                          i,
                          AGG_COMM + world_rank,
                          MPI_COMM_WORLD,
                          &(ph1_request_send[ind_c]));
                ind_c++;
            }
        }
    }

    memset(Y->entries[0], 0,
           Y->m * Y->n * sizeof(double));

    initSendBufferSpace(ph2_send_buffer);
    memset(ph2_send_buffer->data[0], 0, ph2_send_buffer->send_count * ph2_send_buffer->feature_size * sizeof(double));

    MPI_Waitall(ph1_msg_recv_count, ph1_request_recv, MPI_STATUS_IGNORE);
    MPI_Waitall(ph1_msg_send_count, ph1_request_send, MPI_STATUS_IGNORE);


    // COMPUTATION-1
    for(i=0; i<A->m; i++) {
        for (j=A->ia[i]; j<A->ia[i+1]; j++) {
            int target_node = A->ja[j];
            int tmp = A->ja_mapped[j];
            // multip of nodes that already exist in processor
            if (A->inPart[target_node] == world_rank) {
                // local computation (0 to littleM)
                if (i < A->littleM) {
                    for (k = 0; k<Y->n;k++) {
                        Y->entries[i][k] += A->val[j] * X->entries[tmp][k];
                    }
                }
                // partial computation for other processors, thats why results are write down to the phase2's send buffer
                // they will be sent to their processors during post communication
                else {
                    for (k = 0; k<Y->n;k++) {
                        ph2_send_buffer->data[i-A->littleM][k] += A->val[j] * X->entries[tmp][k];
                    }
                }

            }
            // multip of received nodes
            else {
                // if received nodes are the owned nodes of the processor, results will be written to Y
                if (i < A->littleM) {
                    for (k = 0; k<Y->n;k++) {
                        Y->entries[i][k] += A->val[j] * ph1_recv_buffer->data[tmp][k];
                    }
                }
                // if received nodes are used for partial computation, they will be written to phase 2 send_buffer
                // they will be sent during post comm
                else {
                    for (k = 0; k<Y->n;k++) {
                        ph2_send_buffer->data[i - A->littleM][k] += A->val[j] * ph1_recv_buffer->data[tmp][k];
                    }
                }
            }
        }
    }
    
    

    // PHASE-2 COMMUNICATION (POST-COMMUNICATION)
    ind_c = 0;
    initRecvBufferSpace(ph2_recv_buffer);
    for (i = 0;i < world_size; i++) {
        if (i != world_rank) {
            range = ph2_recv_buffer->pid_map[i+1] - ph2_recv_buffer->pid_map[i];
            base = ph2_recv_buffer->pid_map[i];
            if (range != 0) {
                MPI_Irecv(&(ph2_recv_buffer->data[base][0]),
                          range * ph2_recv_buffer->feature_size,
                          MPI_DOUBLE,
                          i,
                          AGG_COMM+i,
                          MPI_COMM_WORLD,
                          &(ph2_request_recv[ind_c]));
                ind_c++;
            }
        }
    }

    ind_c = 0;
    for (i=0;i < world_size; i++) {
        if (i != world_rank) {
            range = ph2_send_buffer->pid_map[i+1] - ph2_send_buffer->pid_map[i];
            base = ph2_send_buffer->pid_map[i];
            if (range != 0) {
                MPI_Isend(&(ph2_send_buffer->data[base][0]),
                          range * ph2_send_buffer->feature_size,
                          MPI_DOUBLE,
                          i,
                          AGG_COMM + world_rank,
                          MPI_COMM_WORLD,
                          &(ph2_request_send[ind_c]));
                ind_c++;
            }
        }
    }
    MPI_Waitall(ph2_msg_recv_count, ph2_request_recv, MPI_STATUS_IGNORE);
    MPI_Waitall(ph2_msg_send_count, ph2_request_send, MPI_STATUS_IGNORE);

    // COMPUTATION-2
    int tmp;
    for (i=0; i<ph2_recv_buffer->recv_count; i++) {
        tmp = ph2_recv_buffer->vertices_local[i];
        for (j = 0; j< ph2_recv_buffer->feature_size; j++) {
            Y->entries[tmp][j] += ph2_recv_buffer->data[i][j];
        }
    }
    

    //free(ph1_request_send);
    //free(ph1_request_recv);
    //free(ph2_request_send);
    //free(ph2_request_recv);
	
	
	
    recvBufferSpaceFree(ph1_recv_buffer);
    sendBufferSpaceFree(ph1_send_buffer);
    recvBufferSpaceFree(ph2_recv_buffer);
    sendBufferSpaceFree(ph2_send_buffer);
    

}


void aggregate_csr(gcnLayer* layer, Matrix* X, Matrix* Y, int step) {
    

}



