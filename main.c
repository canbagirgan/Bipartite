#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "includes/fileio.h"
#include "includes/typedef.h"
#include "includes/basic.h"
#include "includes/gcnLayer.h"
#include "includes/activationLayer.h"
#include "includes/neuralNet.h"
#include "includes/matrix.h"
#include "includes/lossFunctions.h"

#define MAX_LAYER 4

void print_usage() {
    printf("Usage: bin/gcn "    
                   "adj.bin " \
                   "adj_t.bin " \
                   "feature_matrix.csv " \
                   "labels.csv " \
                   "vertex_part.txt " \
                   "[-s isMatrixSymmetric 1 or 0] " \
                   "[-l num_gcn_layer] " \
                   "[-k hidden_size_1 hidden_size_2 ...] " \
                   "[-e num_epoch] \n");
}

int main(int argc, char** argv) {
    printf("hii\n");
    if (argc < 6)
    {
        print_usage();
        return 5;
    }

    // DEFAULT SETTINGS
    int nlayer = 2;
    int k[MAX_LAYER-1];
    k[0] = 10;
    int nepoch = 10;
    int option = 0;
    // END

    for (int i = 6; i < argc; i++) {
        if (strcmp(argv[i], "-k") == 0 && (i + 1) < argc) {
            for (int j = 0; j < nlayer-1; j++){
                k[j] = atoi(argv[i + 1]);
                i++;
            }
        } else if (strcmp(argv[i], "-l") == 0 && (i + 1) < argc) {
            nlayer = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-e") == 0 && (i + 1) < argc) {
            nepoch = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-o") == 0 && (i + 1) < argc) {
            option = atoi(argv[i + 1]);
            i++;
        } else {
            print_usage();
            return 1;
        }
    }
    MPI_Init(&argc, &argv);
    
    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    neural_net* net = net_init(4);
    
    SparseMat* A = readSparseMat(argv[1], STORE_BY_ROWS, argv[5]);
    
    //SparseMat* A_T = readSparseMat(argv[2], STORE_BY_ROWS, argv[5]);
    SparseMat* A_T = A;

    ParMatrix* X = readDenseMat(argv[3], A);
    ParMatrix* Y = readDenseMat(argv[4], A);
    /**
    layer_super gcn_layers[nlayer];
    layer_super activation_layers[nlayer-1];
    */
    
    layer_super* gcn_1 = layer_init_gcn(A, A_T, X->gn, k[0]);
    //layer_super* act_1 = layer_init_activation(RELU);
    //layer_super* gcn_2 = layer_init_gcn(A, A_T, k[0], Y->gn);
    
    net_addLayer(net, gcn_1);
    //net_addLayer(net, act_1);
    //net_addLayer(net, gcn_2);

    Timer time = {9999,9999,9999,9999,9999,9999,9999,9999};
    Stats stats = {0,0,0,0};
    ParMatrix* output;
    
    double start_time, end_time;
    double tot = 0, time1 = 0, time2 = 0, time3 = 0, time4 = 0;

    //for (int c=0; c<5; c++)
    //    output = net_forward(net, X, option, &time);
    
    for(int i = 0; i < nepoch; i++) {
        Matrix* tempErr = matrix_create(A->littleM, Y->gn);

        MPI_Barrier(MPI_COMM_WORLD);
        time1 = MPI_Wtime();
        output = net_forward(net, X, option, &time, &stats);
#ifdef STATS
        MPI_Barrier(MPI_COMM_WORLD);
        time2 = MPI_Wtime();
#endif
        Matrix* soft = matrix_softmax(output->mat);
        //matrix_de_crossEntropy(soft, Y->mat, tempErr);
        //crossEntropy(Y->mat, soft);
        //metrics(soft, Y->mat);
#ifdef STATS
        MPI_Barrier(MPI_COMM_WORLD);
        time3 = MPI_Wtime();
#endif
        //net_backward(net, tempErr, 0.001, i, &time, &stats);

        MPI_Barrier(MPI_COMM_WORLD);
        time4 = MPI_Wtime();
        matrix_free(soft);

        tot += time4 - time1;
    	if (time.totalTime > time4 - time1)
    		time.totalTime = time4 - time1;
#ifdef STATS
        if (time.compTimeF > time2 - time1)
            time.compTimeF = time2 - time1;
        if (time.compTimeB > time4 - time3)
            time.compTimeB = time4 - time3;
        if (time.other > time3 - time2)
            time.other = time3 - time2;
#endif
    }
#ifdef STATS
    int maxSendVol, maxRecvVol, maxSendRecv, maxMultiplyAdd;
    MPI_Reduce(&maxSendVol, &stats.sendVol, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&maxRecvVol, &stats.recvVol, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&maxSendRecv, &stats.sendRecvVol, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&maxMultiplyAdd, &stats.multiplyAdd, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    if (world_rank == 0) {
#ifdef STATS
        time.commTime = time.commTimeB + time.commTimeF;
        time.compTimeF -= time.commTimeF;
        time.compTimeB -= time.commTimeB;
        time.compTime = time.compTimeF + time.compTimeB; 

        printf("2D %d %f %f %f %f %f %f %f %f\n", world_size, time.commTimeF, time.commTimeB, time.commTime, time.compTimeF, time.compTimeB, time.compTime, time.other, time.totalTime);
        //printf("2D %d %f %f %f %f\n", world_size, time.commTime, time.compTime, time.other, time.totalTime);
        //printf("%d %d %d %d\n", maxSendVol/nepoch, maxRecvVol/nepoch, maxSendRecv/nepoch, maxMultiplyAdd/nepoch);
        printf("---------------------------------------------------------------------------------------------\n");
#else
        printf("2D %d %f\n", world_size, time.totalTime);
        //printf("%d %d %d %d\n", maxSendVol/nepoch, maxRecvVol/nepoch, maxSendRecv/nepoch, maxMultiplyAdd/nepoch);        
        //printf("---------------------------------------------------------------------------------------------\n");
#endif
    	//printf("Average runtime for current experiment=> %lf\n", tot / nepoch); 
    	//printf("Min runtime for current experiment=> %lf\n", min);
    	//printf("---------------------------------------------\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);

    //bufferFree(gcn_1->gcn_layer->p1_recvBuffer_F);
    //bufferFree(gcn_1->gcn_layer->p2_recvBuffer_F);
    //bufferFree(gcn_1->gcn_layer->p1_recvBuffer_B);
    //bufferFree(gcn_1->gcn_layer->p2_recvBuffer_B);
    //bufferFree(gcn_1->gcn_layer->p1_sendBuffer_F);
    //bufferFree(gcn_1->gcn_layer->p2_sendBuffer_F);
    //bufferFree(gcn_1->gcn_layer->p1_sendBuffer_B);
    //bufferFree(gcn_1->gcn_layer->p2_sendBuffer_B);
  
    //net_free(net);
    MPI_Finalize();
    return 0;
}