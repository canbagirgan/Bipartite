#ifndef BASIC_H_INCLUDED
#define BASIC_H_INCLUDED

#pragma once
#include "typedef.h"

//int mapG2L(int i, int offset);
//int mapProcessor(int i);
//sendTable* initSendTable(SparseMat* A);
//recvTable* initRecvTable(SparseMat* A_T);
//sendBuffer* initSendBuffer(SparseMat* A, int *l2gMap, int feature_size);
//recvBuffer* initRecvBuffer(SparseMat* A, int feature_size, sendBuffer* sendBuff);
Buffer* initSendBuffer(SparseMat* A, int *l2gMap, int feature_size);
Buffer* initRecvBuffer(SparseMat* A, int feature_size, Buffer* sendBuff);
#endif // BASIC_H_INCLUDED
