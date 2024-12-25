#ifndef GCNLAYER_H_INCLUDED
#define GCNLAYER_H_INCLUDED
#include "typedef.h"
#include "basic.h"


#define FORWARD 101
#define BACKWARD 102

typedef struct {
    int size_n; // nxn matrisin ni
    int size_f; // layerın feature_size'ı
    int size_output;
    SparseMat* adjacency;
    SparseMat* adjacency_T; //edge partitionda local transpose alınabilir(farklı dosyadan okumaya gerek olmayabilir)
    // Yapamadık usta farklı dosyaya gerek oldu
    ParMatrix* input; //H(l-1)
    ParMatrix* output; //H(l)
    Matrix* weights; 
    Matrix* m_weights; //backprop related (adam)
    Matrix* v_weights; //backprop related (adam)
    Matrix* gradients; 
    //sendBuffer* ph2_sendBuffer_forward;
    //recvBuffer* ph2_recvBuffer_forward;
    //sendBuffer* ph2_sendBuffer_backward;
    //recvBuffer* ph2_recvBuffer_backward;
    //sendBuffer* ph1_sendBuffer_forward;
    //recvBuffer* ph1_recvBuffer_forward;
    //sendBuffer* ph1_sendBuffer_backward;
    //recvBuffer* ph1_recvBuffer_backward;
    Buffer *p1_sendBuffer_F;
    Buffer *p2_sendBuffer_F;
    Buffer *p1_recvBuffer_F;
    Buffer *p2_recvBuffer_F;
    Buffer *p1_sendBuffer_B;
    Buffer *p2_sendBuffer_B;
    Buffer *p1_recvBuffer_B;
    Buffer *p2_recvBuffer_B;
    int p1_send_cnt_F;
    int p2_send_cnt_F;
    int p1_recv_cnt_F;
    int p2_recv_cnt_F;
    int p1_send_cnt_B;
    int p2_send_cnt_B;
    int p1_recv_cnt_B;
    int p2_recv_cnt_B;
    //int ph1_msgSendCount_forward;
    //int ph1_msgRecvCount_forward;
    //int ph1_msgSendCount_backward;
    //int ph1_msgRecvCount_backward;
    //int ph2_msgSendCount_forward;
    //int ph2_msgRecvCount_forward;
    //int ph2_msgSendCount_backward;
    //int ph2_msgRecvCount_backward;
} gcnLayer;

gcnLayer* gcn_init(SparseMat* adj, SparseMat* adj_T, int size_f,int size_out, int isDirected);
void setMode(int i);
void gcn_forward(gcnLayer* layer, int option, Timer *time, Stats *stats);
Matrix* gcn_backward(gcnLayer* layer, Matrix* out_error, Timer *time, Stats *stats);
void gcn_step(gcnLayer* layer, double lr, int t);
void gcn_free(gcnLayer* layer);
#endif // GCNLAYER_H_INCLUDED
