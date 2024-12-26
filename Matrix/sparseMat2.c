#include "../includes/sparseMat.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

#ifdef ISEND
#define _SEND(buff, req, tag) MPI_Isend(&(buff->data[base][0]), range * buff->feature_size, MPI_DOUBLE, trgt, tag, MPI_COMM_WORLD, req)
#else
#define _SEND(buff, req, tag) MPI_Send(&(buff->data[base][0]), range * buff->feature_size, MPI_DOUBLE, trgt, tag, MPI_COMM_WORLD)
#endif

#ifdef IRECV
#define _RECV(buff, req, tag) MPI_Irecv(&(buff->data[base][0]), range * buff->feature_size, MPI_DOUBLE, trgt, tag, MPI_COMM_WORLD, req)
#else
#define _RECV(buff, req, tag) MPI_Recv(&(buff->data[base][0]), range * buff->feature_size, MPI_DOUBLE, trgt, tag, MPI_COMM_WORLD)
#endif

void aggregate(gcnLayer* layer, Matrix* X, Matrix* Y, int step, Timer *time, Stats *stats) {
    stats->sendVol = 0;
    stats->recvVol = 0;
    stats->sendRecvVol = 0;
    stats->multiplyAdd = 0;
    int world_size, world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    SparseMat* A;
    int i = 0, j = 0, k = 0;
    Buffer *ph1_send_buffer, *ph1_recv_buffer, *ph2_send_buffer, *ph2_recv_buffer;
    int ph1_msg_send_count = 0, ph1_msg_recv_count = 0, ph2_msg_send_count = 0, ph2_msg_recv_count = 0;
    int ind = 0;
    int range = 0, base = 0;
    int target_node = 0, l_id_target_node = 0, tmp = 0;
    int trgt = 0;

    if (step == FORWARD) {
        A = layer->adjacency;
        ph1_send_buffer = layer->p1_sendBuffer_F;
        ph1_recv_buffer = layer->p1_recvBuffer_F;
        ph2_send_buffer = layer->p2_sendBuffer_F;
        ph2_recv_buffer = layer->p2_recvBuffer_F;

        ph1_msg_send_count = layer->p1_send_cnt_F;
        ph2_msg_send_count = layer->p2_send_cnt_F;
        ph1_msg_recv_count = layer->p1_recv_cnt_F;
        ph2_msg_recv_count = layer->p2_recv_cnt_F;
    }
    else if (step == BACKWARD) {
        A = layer->adjacency;
        ph1_send_buffer = layer->p1_sendBuffer_B;
        ph1_recv_buffer = layer->p1_recvBuffer_B;
        ph2_send_buffer = layer->p2_sendBuffer_B;
        ph2_recv_buffer = layer->p2_recvBuffer_B;

        ph1_msg_send_count = layer->p1_send_cnt_B;
        ph2_msg_send_count = layer->p2_send_cnt_B;
        ph1_msg_recv_count = layer->p1_recv_cnt_B;
        ph2_msg_recv_count = layer->p2_recv_cnt_B;
    }

    memset(ph2_send_buffer->data[0], 0, ph2_send_buffer->count * ph2_send_buffer->feature_size * sizeof(double));
    
    // ----------- START -------------
    
    // PH1 RECV
    for (i = 0; i < ph1_msg_recv_count; i++) {
        trgt = ph1_recv_buffer->list[i];
        range = ph1_recv_buffer->pid_map[trgt+1] - ph1_recv_buffer->pid_map[trgt];
        base = ph1_recv_buffer->pid_map[trgt];
        _RECV(ph1_recv_buffer, &(ph1_recv_buffer->reqs[i]), AGG_COMM + trgt);
        stats->recvVol += range;
    }
    // PH2 RECV
    for (i = 0; i < ph2_msg_recv_count; i++) {
        trgt = ph2_recv_buffer->list[i];
        range = ph2_recv_buffer->pid_map[trgt+1] - ph2_recv_buffer->pid_map[trgt];
        base = ph2_recv_buffer->pid_map[trgt];
        _RECV(ph2_recv_buffer, &(ph2_recv_buffer->reqs[i]), AGG_COMM + trgt);
        stats->recvVol += range;
    }
    
    // PH1 SEND
    for (i = 0; i < ph1_msg_send_count; i++) {
        trgt = ph1_send_buffer->list[i];
        range = ph1_send_buffer->pid_map[trgt+1] - ph1_send_buffer->pid_map[trgt];
        base = ph1_send_buffer->pid_map[trgt];

        for (j = 0; j < range; j++) {
            ind = ph1_send_buffer->vertices_local[base + j];
            memcpy(ph1_send_buffer->data[base + j],  X->entries[ind] , sizeof(double) * ph1_send_buffer->feature_size);
        }
        _SEND(ph1_send_buffer, &(ph1_send_buffer->reqs[i]), AGG_COMM + world_rank);
        stats->sendVol += range;
    }   

#ifdef IRECV
    MPI_Waitall(ph1_msg_recv_count, ph1_recv_buffer->reqs, MPI_STATUS_IGNORE);
#endif
    // COMPUTATION - 1
    for(i = 0; i < A->m; i++) {
        for (j = A->ia[i]; j < A->ia[i+1]; j++) {
            target_node = A->ja[j];
            l_id_target_node = A->ja_mapped[j];
            if (A->inPart[target_node] == world_rank)
            {
                if (i < A->littleM) 
                {
                    for (k = 0; k < Y->n; k++) {
                        Y->entries[i][k] += A->val[j] * X->entries[l_id_target_node][k];
                        stats->multiplyAdd += 1;
                    }
                } 
                else {
                    for (k = 0; k < Y->n; k++) {
                        ph2_send_buffer->data[i - A->littleM][k] += A->val[j] * X->entries[l_id_target_node][k];
                        stats->multiplyAdd += 1;
                    }
                }     
            } 
            else {
                for (k = 0; k < Y->n; k++) {
                    Y->entries[i][k] += A->val[j] * ph1_recv_buffer->data[l_id_target_node][k];
                    stats->multiplyAdd += 1;
                }
            } 
        }
    }

    // PH2 SEND
    for (i = 0; i < ph2_msg_send_count; i++) {
        trgt = ph2_send_buffer->list[i];
        range = ph2_send_buffer->pid_map[trgt+1] - ph2_send_buffer->pid_map[trgt];
        base = ph2_send_buffer->pid_map[trgt];
        _SEND(ph2_send_buffer, &(ph2_send_buffer->reqs[i]), AGG_COMM + world_rank);
        stats->sendVol += range;
    }

    stats->sendRecvVol = stats->sendVol + stats->recvVol;

#ifdef IRECV
    MPI_Waitall(ph2_msg_recv_count, ph2_recv_buffer->reqs, MPI_STATUS_IGNORE);
#endif
    // SUMMATION OF THE PARTIAL RESULTS
    for (i = 0; i < ph2_recv_buffer->count; i++) {
        tmp = ph2_recv_buffer->vertices_local[i];
        for (j = 0; j < ph2_recv_buffer->feature_size; j++) {
            Y->entries[tmp][j] += ph2_recv_buffer->data[i][j];
        }
    }

#ifdef ISEND
    MPI_Waitall(ph2_msg_send_count, ph2_send_buffer->reqs, MPI_STATUS_IGNORE);
    MPI_Waitall(ph1_msg_send_count, ph1_send_buffer->reqs, MPI_STATUS_IGNORE);
#endif

}

void aggregate_shuffled(gcnLayer* layer, Matrix* X, Matrix* Y, int step, Timer *time, Stats *stats) {
    int world_size, world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    SparseMat* A;
    int i = 0, j = 0, k = 0;
    Buffer *ph1_send_buffer, *ph1_recv_buffer, *ph2_send_buffer, *ph2_recv_buffer;
    int ph1_msg_send_count = 0, ph1_msg_recv_count = 0, ph2_msg_send_count = 0, ph2_msg_recv_count = 0;
    int ind = 0;
    int range = 0, base = 0;
    int target_node = 0, l_id_target_node = 0, tmp = 0;
    int trgt = 0;

    if (step == FORWARD) {
        A = layer->adjacency;
        ph1_send_buffer = layer->p1_sendBuffer_F;
        ph1_recv_buffer = layer->p1_recvBuffer_F;
        ph2_send_buffer = layer->p2_sendBuffer_F;
        ph2_recv_buffer = layer->p2_recvBuffer_F;

        ph1_msg_send_count = layer->p1_send_cnt_F;
        ph2_msg_send_count = layer->p2_send_cnt_F;
        ph1_msg_recv_count = layer->p1_recv_cnt_F;
        ph2_msg_recv_count = layer->p2_recv_cnt_F;
    }
    else if (step == BACKWARD) {
        A = layer->adjacency;
        ph1_send_buffer = layer->p1_sendBuffer_B;
        ph1_recv_buffer = layer->p1_recvBuffer_B;
        ph2_send_buffer = layer->p2_sendBuffer_B;
        ph2_recv_buffer = layer->p2_recvBuffer_B;

        ph1_msg_send_count = layer->p1_send_cnt_B;
        ph2_msg_send_count = layer->p2_send_cnt_B;
        ph1_msg_recv_count = layer->p1_recv_cnt_B;
        ph2_msg_recv_count = layer->p2_recv_cnt_B;
    }
    //memset(Y->entries[0], 0, Y->m * Y->n * sizeof(double));
    memset(ph2_send_buffer->data[0], 0, ph2_send_buffer->count * ph2_send_buffer->feature_size * sizeof(double));
    int sendvol1 = 0, sendvol2 = 0;
    
    // ----------- START -------------

    // PH2 Send Buffer Filling Calculation
    for(i = A->littleM; i < A->m; i++) {
        for (j = A->ia[i]; j < A->ia[i+1]; j++) {
            target_node = A->ja[j];
            l_id_target_node = A->ja_mapped[j];
            if (A->inPart[target_node] == world_rank)
            {
                for (k = 0; k < Y->n; k++){
                    ph2_send_buffer->data[i - A->littleM][k] += A->val[j] * X->entries[l_id_target_node][k];
                    stats->multiplyAdd++;
                }
            } 
        }
    }

    // PH2 RECV
    for (i = 0; i < ph2_msg_recv_count; i++) {
        trgt = ph2_recv_buffer->list[i];
        range = ph2_recv_buffer->pid_map[trgt+1] - ph2_recv_buffer->pid_map[trgt];
        base = ph2_recv_buffer->pid_map[trgt];

        _RECV(ph2_recv_buffer, &(ph2_recv_buffer->reqs[i]), AGG_COMM + trgt);

    }
    // PH1 RECV - PH2 SEND

    for (i = 0; i < ph1_msg_recv_count; i++) {
        trgt = ph1_recv_buffer->list[i];
        range = ph1_recv_buffer->pid_map[trgt+1] - ph1_recv_buffer->pid_map[trgt];
        base = ph1_recv_buffer->pid_map[trgt];
        _RECV(ph1_recv_buffer, &(ph1_recv_buffer->reqs[i]), AGG_COMM + trgt);
        _SEND(ph2_send_buffer, &(ph2_send_buffer->reqs[i]), AGG_COMM + world_rank);
    }

    // PH1 SEND
    for (i = 0; i < ph1_msg_send_count; i++) {
        trgt = ph1_send_buffer->list[i];
        range = ph1_send_buffer->pid_map[trgt+1] - ph1_send_buffer->pid_map[trgt];
        base = ph1_send_buffer->pid_map[trgt];

        for (j = 0; j < range; j++) {
            ind = ph1_send_buffer->vertices_local[base + j];
            memcpy(ph1_send_buffer->data[base + j],  X->entries[ind] , sizeof(double) * ph1_send_buffer->feature_size);
        }
        _SEND(ph1_send_buffer, &(ph1_send_buffer->reqs[i]), AGG_COMM + world_rank);
    }

    MPI_Waitall(ph1_msg_recv_count, ph1_recv_buffer->reqs, MPI_STATUS_IGNORE);

    // COMPUTATION - 1
    for(i = 0; i < A->littleM; i++) {
        for (j = A->ia[i]; j < A->ia[i+1]; j++) {
            target_node = A->ja[j];
            l_id_target_node = A->ja_mapped[j];
            if (A->inPart[target_node] == world_rank)
            {
                for (k = 0; k < Y->n; k++) {
                    Y->entries[i][k] += A->val[j] * X->entries[l_id_target_node][k];
                    stats->multiplyAdd++;
                }
            } else {
                for (k = 0; k < Y->n; k++) {
                    Y->entries[i][k] += A->val[j] * ph1_recv_buffer->data[l_id_target_node][k];
                    stats->multiplyAdd++;
                }    
            }
        }
    }

    MPI_Waitall(ph2_msg_recv_count, ph2_recv_buffer->reqs, MPI_STATUS_IGNORE);

    // SUMMATION OF THE PARTIAL RESULTS
    for (i = 0; i < ph2_recv_buffer->count; i++) {
        tmp = ph2_recv_buffer->vertices_local[i];
        for (j = 0; j < ph2_recv_buffer->feature_size; j++) {
            Y->entries[tmp][j] += ph2_recv_buffer->data[i][j];
        }
    }

#ifdef ISEND
    MPI_Waitall(ph2_msg_send_count, ph2_send_buffer->reqs, MPI_STATUS_IGNORE);
    MPI_Waitall(ph1_msg_send_count, ph1_send_buffer->reqs, MPI_STATUS_IGNORE);
#endif

}