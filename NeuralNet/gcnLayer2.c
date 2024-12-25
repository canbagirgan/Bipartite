#include "../includes/gcnLayer.h"
#include "../includes/sparseMat.h"
#include "../includes/matrix.h"
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

int MODE = 0;

/*
Function is ready to be intergrated into sparse_Mat struct
Then use implement aggregation function
Don't forget memory management
*/


gcnLayer* gcn_init(SparseMat* adj, SparseMat* adj_T, int size_f, int size_out, int isDirected) {

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int i, temp;
    gcnLayer* layer = (gcnLayer*) malloc(sizeof(gcnLayer));
    layer->size_n = adj->littleM;
    layer->size_f = size_f;
    layer->size_output= size_out;
    layer->adjacency = adj;
    layer->adjacency_T = adj_T;

    layer->p1_recvBuffer_F = initSendBuffer(adj_T, adj_T->l2gMap, size_f);
    layer->p1_sendBuffer_F = initRecvBuffer(adj_T, size_f, layer->p1_recvBuffer_F);
    layer->p2_sendBuffer_F = initSendBuffer(adj, adj->l2gMap, size_f);
    layer->p2_recvBuffer_F = initRecvBuffer(adj, size_f, layer->p2_sendBuffer_F);

    layer->p1_recvBuffer_B = initSendBuffer(adj_T, adj_T->l2gMap, size_f);
    layer->p1_sendBuffer_B = initRecvBuffer(adj_T, size_f, layer->p1_recvBuffer_B);
    layer->p2_sendBuffer_B = initSendBuffer(adj, adj->l2gMap, size_f);
    layer->p2_recvBuffer_B = initRecvBuffer(adj, size_f, layer->p2_sendBuffer_B);

    initBufferSpace(layer->p1_recvBuffer_F);
    initBufferSpace(layer->p1_sendBuffer_F);
    initBufferSpace(layer->p2_recvBuffer_F);
    initBufferSpace(layer->p2_sendBuffer_F);
    initBufferSpace(layer->p1_recvBuffer_B);
    initBufferSpace(layer->p1_sendBuffer_B);
    initBufferSpace(layer->p2_recvBuffer_B);
    initBufferSpace(layer->p2_sendBuffer_B);


    int* g2l_p1 = (int *) malloc(sizeof(int) * adj->gm);
    int* g2l_p2 = (int *) malloc(sizeof(int) * adj_T->gm);
    memset(g2l_p1, -1, sizeof(int) * adj->gm);
    memset(g2l_p2, -1, sizeof(int) * adj_T->gm);
 

    for (i=0; i < adj->littleM;i++) {
		temp = adj->l2gMap[i];
        g2l_p1[temp] = i;
	}
    // recv
    for (i=0;i<layer->p1_recvBuffer_F->count;i++) {
		temp = layer->p1_recvBuffer_F->vertices[i];
        g2l_p1[temp] = i;
	}

    for (i=0; i < adj_T->littleM;i++) {
		temp = adj_T->l2gMap[i];
        g2l_p2[temp] = i;
	}
    
    // recv
    for (i=0;i<layer->p2_sendBuffer_F->count;i++) {
		temp = layer->p2_sendBuffer_F->vertices[i];
        g2l_p2[temp] = i;
	}
    
    int* vertices_local_p1_recv = (int *) malloc(sizeof(int) * layer->p1_recvBuffer_F->count);
    int* vertices_local_p1_send = (int *) malloc(sizeof(int) * layer->p1_sendBuffer_F->count);
    int* vertices_local_p2_recv = (int *) malloc(sizeof(int) * layer->p2_recvBuffer_F->count);
    int* vertices_local_p2_send = (int *) malloc(sizeof(int) * layer->p2_sendBuffer_F->count);

    if (adj->init == 0) {
        for (i=0;i < adj->nnz; i++) {
    		temp = adj->ja[i];
    		adj->ja_mapped[i] = g2l_p1[temp];
    	}
        adj->init = 1;
    }
    for (i=0; i<layer->p1_recvBuffer_F->count; i++) {
            temp = layer->p1_recvBuffer_F->vertices[i];
            vertices_local_p1_recv[i] = g2l_p1[temp];
    }
    for (i=0; i<layer->p1_sendBuffer_F->count; i++) {
            temp = layer->p1_sendBuffer_F->vertices[i];
            vertices_local_p1_send[i] = g2l_p1[temp];
    }

    if (adj_T->init == 0) {
        for (i=0;i < adj_T->nnz; i++) {
    		temp = adj_T->ja[i];
    		adj_T->ja_mapped[i] = g2l_p2[temp];
    	}
        
        adj_T->init = 1;
    }
    for (i=0; i<layer->p2_recvBuffer_F->count; i++) {
            temp = layer->p2_recvBuffer_F->vertices[i];
            vertices_local_p2_recv[i] = g2l_p2[temp];
    }
    for (i=0; i<layer->p2_sendBuffer_F->count; i++) {
            temp = layer->p2_sendBuffer_F->vertices[i];
            vertices_local_p2_send[i] = g2l_p2[temp];
    }

    layer->p1_recvBuffer_F->vertices_local = vertices_local_p1_recv;
    layer->p1_sendBuffer_F->vertices_local = vertices_local_p1_send;
    layer->p2_recvBuffer_F->vertices_local = vertices_local_p2_recv;    
    layer->p2_sendBuffer_F->vertices_local = vertices_local_p2_send;

    layer->p1_recvBuffer_B->vertices_local = vertices_local_p1_recv;
    layer->p1_sendBuffer_B->vertices_local = vertices_local_p1_send;
    layer->p2_recvBuffer_B->vertices_local = vertices_local_p2_recv;    
    layer->p2_sendBuffer_B->vertices_local = vertices_local_p2_send;

    // defining local-local vertices ph1send-ph2recv for gemm overlap
    int *ll_mask = (int *) calloc(adj->littleM, sizeof(int));
    
    for (i=0; i<layer->p1_sendBuffer_F->count; i++){
        ll_mask[layer->p1_sendBuffer_F->vertices_local[i]] = 1;
    }
    layer->p1_sendBuffer_F->ll_count = 0;
    for (i=0; i<adj->littleM; i++) {
        if (ll_mask[i] == 0) layer->p1_sendBuffer_F->ll_count++;
    }
    layer->p1_sendBuffer_F->ll_vertices = (int *) malloc(sizeof (int) * layer->p1_sendBuffer_F->ll_count);
    int idx = 0;
    for (i=0; i<adj->littleM; i++) {
        if (ll_mask[i] == 0) {
            layer->p1_sendBuffer_F->ll_vertices[idx++] = i;
        }
    } 
    free(ll_mask);
    ll_mask = NULL;

    layer->p1_send_cnt_F = 0;
    layer->p1_recv_cnt_F = 0;
    for (i=0;i < world_size; i++) {
    	int range = layer->p1_sendBuffer_F->pid_map[i+1] - layer->p1_sendBuffer_F->pid_map[i];
        int rRange = layer->p1_recvBuffer_F->pid_map[i+1] - layer->p1_recvBuffer_F->pid_map[i];
        if (i != world_rank) {
            if (range != 0) {
                layer->p1_send_cnt_F++;
            }
            if (rRange != 0) {
                layer->p1_recv_cnt_F++;
            }
        }
    }

    layer->p2_send_cnt_F = 0;
    layer->p2_recv_cnt_F = 0;
    for (i=0;i < world_size; i++) {
    	int range = layer->p2_sendBuffer_F->pid_map[i+1] - layer->p2_sendBuffer_F->pid_map[i];
        int rRange = layer->p2_recvBuffer_F->pid_map[i+1] - layer->p2_recvBuffer_F->pid_map[i];
        if (i != world_rank) {
            if (range != 0) {
                layer->p2_send_cnt_F++;
            }
            if (rRange != 0) {
                layer->p2_recv_cnt_F++;
            }
        }
    }
    
    layer->p1_sendBuffer_F->list = (int *) malloc(sizeof(int) * layer->p1_send_cnt_F);
    layer->p1_recvBuffer_F->list = (int *) malloc(sizeof(int) * layer->p1_recv_cnt_F);
    layer->p2_sendBuffer_F->list = (int *) malloc(sizeof(int) * layer->p2_send_cnt_F);
    layer->p2_recvBuffer_F->list = (int *) malloc(sizeof(int) * layer->p2_recv_cnt_F);

    int ctr = 0, ctr_r = 0;
    for (i=0;i < world_size; i++) {
    	int range = layer->p1_sendBuffer_F->pid_map[i+1] - layer->p1_sendBuffer_F->pid_map[i];
        int rRange = layer->p1_recvBuffer_F->pid_map[i+1] - layer->p1_recvBuffer_F->pid_map[i];
        if (i != world_rank) {
            if (range != 0) {
                layer->p1_sendBuffer_F->list[ctr] = i;
                ctr++;
            }
            if (rRange != 0) {
                layer->p1_recvBuffer_F->list[ctr_r] = i;
                ctr_r++;
            }
        }
    }

    ctr = 0, ctr_r = 0;
    for (i=0;i < world_size; i++) {
    	int range = layer->p2_sendBuffer_F->pid_map[i+1] - layer->p2_sendBuffer_F->pid_map[i];
        int rRange = layer->p2_recvBuffer_F->pid_map[i+1] - layer->p2_recvBuffer_F->pid_map[i];
        if (i != world_rank) {
            if (range != 0) {
                layer->p2_sendBuffer_F->list[ctr] = i;
                ctr++;
            }
            if (rRange != 0) {
                layer->p2_recvBuffer_F->list[ctr_r] = i;
                ctr_r++;
            }
        }
    }
    layer->p1_sendBuffer_B->list = layer->p1_sendBuffer_F->list;
    layer->p1_recvBuffer_B->list = layer->p1_recvBuffer_F->list;
    layer->p2_sendBuffer_B->list = layer->p2_sendBuffer_F->list;
    layer->p2_recvBuffer_B->list = layer->p2_recvBuffer_F->list;
    
    layer->p1_send_cnt_B = layer->p1_send_cnt_F;
    layer->p1_recv_cnt_B = layer->p1_recv_cnt_F;
    layer->p2_send_cnt_B = layer->p2_send_cnt_F;
    layer->p2_recv_cnt_B = layer->p2_recv_cnt_F;

#ifdef DEBUG
    if (world_rank == 40) {
        
        printf("P1 Recv Volume: %d\n", layer->p1_recvBuffer_F->count);
        printf("P1 Send Volume: %d\n", layer->p1_sendBuffer_F->count);
        printf("P2 Recv Volume: %d\n", layer->p2_recvBuffer_F->count);
        printf("P2 Send Volume: %d\n", layer->p2_sendBuffer_F->count);
         
        printf("P1 Recv buffer: ");
        for (int i=0; i<layer->p1_recvBuffer_F->count; i++){
            printf("%d ", layer->p1_recvBuffer_F->vertices[i]);
        }
        printf("\nP1 Send buffer: ");
        for (int i=0; i<layer->p1_sendBuffer_F->count; i++){
            printf("%d ", layer->p1_sendBuffer_F->vertices[i]);
        }
        
        printf("\nP2 Recv buffer: ");
        for (int i=0; i<layer->p2_recvBuffer_F->count; i++){
            printf("%d ", layer->p2_recvBuffer_F->vertices[i]);
        }
        
        printf("\n\nP2 Send buffer: ");
        for (int i=0; i<layer->p2_sendBuffer_F->count; i++){
            printf("%d ", layer->p2_sendBuffer_F->vertices[i]);
        }
        /**
        printf("\n\nP2 Recv buffer PIDMAP: ");
        for (int i=0; i<world_size+1; i++){
            printf("%d ", layer->p2_recvBuffer->pid_map[i]);
        }

        printf("\n\nP2 Send buffer PIDMAP: ");
        for (int i=0; i<world_size+1; i++){
            printf("%d ", layer->p2_sendBuffer->pid_map[i]);
        }
        
        printf("P1 Recv G2L: ");
        for (int i=0; i<adj_T->gm; i++){
            printf("%d ", g2l_p1_recv[i]);
        }
        printf("\nP1 Send G2L: ");
        for (int i=0; i<adj_T->gm; i++){
            printf("%d ", g2l_p1_send[i]);
        }
        printf("\nP2 Recv G2L: ");
        for (int i=0; i<adj->gm; i++){
            printf("%d ", g2l_p2_recv[i]);
        }
        printf("\nP2 Send G2L: ");
        for (int i=0; i<adj->gm; i++){
            printf("%d ", g2l_p2_send[i]);
        }
        */
        printf("\n\nP1 G2L: ");
        for (int i=0; i<adj_T->gm; i++){
            printf("%d ", g2l_p1[i]);
        }
        printf("\n\nP2 G2L: ");
        for (int i=0; i<adj->gm; i++){
            printf("%d ", g2l_p2[i]);
        }
        printf("\n\nTHE G2L: ");
        for (int i=0; i<adj->gm; i++){
            printf("%d ", g2l[i]);
        }
        
        printf("\nAdj-T nnz: %d", adj_T->nnz);
        printf("\nAdj nnz: %d", adj->nnz);
        printf("\nAdj JA: ");
        for (i=0;i < adj->nnz; i++){
    		temp = adj->ja[i];
            printf("%d ", temp);
        }
        printf("\nAdj-T JA: ");
        for (i=0;i < adj_T->nnz; i++) {
    		temp = adj_T->ja[i];
            printf("%d ", temp);
        }
        
        printf("\nP1 Recv VL: ");
        for (i=0; i<layer->p1_recvBuffer_F->count; i++) {
            temp = layer->p1_recvBuffer_F->vertices_local[i];
            printf("%d ", temp);
        }
        printf("\nP1 Send VL: ");
        for (i=0; i<layer->p1_sendBuffer_F->count; i++) {
            temp = layer->p1_sendBuffer_F->vertices_local[i];
            printf("%d ", temp);
        }
        printf("\nP2 Recv VL: ");
        for (i=0; i<layer->p2_recvBuffer_F->count; i++) {
            temp = layer->p2_recvBuffer_F->vertices_local[i];
            printf("%d ", temp);
        }
        printf("\nP2 Send VL: ");
        for (i=0; i<layer->p2_sendBuffer_F->count; i++) {
            temp = layer->p2_sendBuffer_F->vertices_local[i];
            printf("%d ", temp);
        }
        printf("\nAdj JAmapped: ");
        for (i=0;i < adj->nnz; i++) {
    		printf("%d ", adj->ja_mapped[i]);
    	}
        printf("\nAdj-T JAmapped: ");
        for (i=0;i < adj_T->nnz; i++) {
    		printf("%d ", adj_T->ja_mapped[i]);
    	}
        printf("\nTHE JAmapped: ");
        for (i=0;i < adj->nnz; i++) {
    		printf("%d ", ja_mapped[i]);
    	}
        printf("\nP1 Recv Count: %d\n", layer->p1_recv_cnt_F);
        printf("P1 Send Count: %d\n", layer->p1_send_cnt_F);
        printf("P2 Recv Count: %d\n", layer->p2_recv_cnt_F);
        printf("P2 Send Count: %d\n", layer->p2_send_cnt_F);

        printf("\n\nP1 Recv buffer PIDMAP: ");
        for (int i=0; i<world_size+1; i++){
            printf("%d ", layer->p1_recvBuffer_F->pid_map[i]);
        }
        printf("\n\nP1 Send buffer PIDMAP: ");
        for (int i=0; i<world_size+1; i++){
            printf("%d ", layer->p1_sendBuffer_F->pid_map[i]);
        }
        printf("\n\nP2 Recv buffer PIDMAP: ");
        for (int i=0; i<world_size+1; i++){
            printf("%d ", layer->p2_recvBuffer_F->pid_map[i]);
        }
        printf("\n\nP2 Send buffer PIDMAP: ");
        for (int i=0; i<world_size+1; i++){
            printf("%d ", layer->p2_sendBuffer_F->pid_map[i]);
        }
        printf("\n");
    }
#endif
    free(g2l_p1);
    free(g2l_p2);


    layer->output = (ParMatrix*) malloc(sizeof(ParMatrix));
    layer->output->gm = adj->gm;
    layer->output->gn = size_out;
    layer->output->store = adj->store;
    layer->output->l2gMap = adj->l2gMap;
    layer->output->inPart = adj->inPart;
    layer->output->mat = matrix_create(adj->littleM, size_out);

    if (world_rank == 0) {
        init_weights_random(layer, 10);
    } else {
        layer->weights = matrix_create(layer->size_f, layer->size_output);
    }
    // eğer canım sıkılırsa ki sıkılmaz weight matrisi de paralel okunup sonra allreduce edilebilir
    MPI_Bcast(&(layer->weights->entries[0][0]), layer->weights->m * layer->weights->n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    layer->gradients = matrix_create(layer->size_f, layer->size_output);
    layer->m_weights = matrix_create(layer->size_f, layer->size_output);
    matrix_fill(layer->m_weights, 0);
    layer->v_weights = matrix_create(layer->size_f, layer->size_output);
    matrix_fill(layer->v_weights, 0);

    
    return layer;
}

void setMode(int i) {
    MODE = i;
}

void gcn_forward(gcnLayer* layer, int option, Timer *time, Stats *stats) {
    Matrix* temp = matrix_create(layer->size_n, layer->size_f);
    switch (option)
    {
        case 0:
            aggregate(layer, layer->input->mat, temp, FORWARD, time, stats);
            break;
        default:
            break;
    }
    GEMM(temp, layer->weights, layer->output->mat);
    matrix_free(temp);
}

Matrix* gcn_backward(gcnLayer* layer, Matrix* out_error, Timer *time, Stats *stats) {
    Matrix* temp = matrix_create(layer->size_n, layer->size_output);
    
    Matrix* out = matrix_create(layer->size_n, layer->size_f);
	
    aggregate(layer, out_error, temp, BACKWARD, time, stats);
    
    GEMM_NT(temp, layer->weights, out);
    
    GEMM_TN(layer->input->mat, temp, layer->gradients);
    
    matrix_free(temp);
    //printf("flag_4 \n");
    return out;
}


//Later change this as adam and generate normal
void gcn_step(gcnLayer* layer, double lr, int t) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    double eta = 0.01;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 0.00000001;
    Matrix* temp = matrix_create(layer->gradients->m, layer->gradients->n);
    MPI_Allreduce(&(layer->gradients->entries[0][0]), &(temp->entries[0][0]),
                   layer->gradients->m * layer->gradients->n,
                   MPI_DOUBLE,
                   MPI_SUM,
                   MPI_COMM_WORLD);
    //printf("%lf \n", layer->gradients->entries[0][0]);
    matrix_scale(lr, temp);

    matrix_scale(beta1, layer->m_weights);
    Matrix* temp_m = matrix_scale_return((1-beta1), temp);
    matrix_sum(layer->m_weights, temp_m, layer->m_weights);

    //matrix_free(temp_m);

    matrix_scale(beta2, layer->v_weights);

    matrix_multiply(temp, temp, temp_m);

    matrix_scale((1-beta2), temp_m);
    matrix_sum(layer->v_weights, temp_m, layer->v_weights);

    matrix_free(temp_m);

    //bias correction can go here
    Matrix* m_dw_corr = matrix_scale_return(1 / (1 - pow(beta1, t + 1)), layer->m_weights);
    Matrix* v_dw_corr = matrix_scale_return(1 / (1 - pow(beta2, t + 1)), layer->v_weights);


    Matrix* tmp_sqrt = matrix_sqrt(v_dw_corr);
    matrix_addScalar(tmp_sqrt, epsilon);
    matrix_divide(m_dw_corr, tmp_sqrt, tmp_sqrt);
    matrix_scale(eta, tmp_sqrt);



    matrix_subtract(layer->weights, tmp_sqrt, layer->weights);


    //printf("%lf \n", layer->weights->entries[0][0]);
    matrix_free(temp);
    matrix_free(tmp_sqrt);
    matrix_free(m_dw_corr);
    matrix_free(v_dw_corr);
}

void gcn_free(gcnLayer* layer) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	/*
    sparseMatFree(layer->adjacency);
    sparseMatFree(layer->adjacency_T);
    sendBufferListFree(layer->sendBuffer,world_size, world_rank);
    recvBufferListFree(layer->recvBufferList,world_size, world_rank);
    sendBufferListFree(layer->sendBuffer_backward,world_size, world_rank);
    recvBufferListFree(layer->recvBufferList_backward,world_size, world_rank);
    free(layer);
    */
    layer = NULL;
    
}
