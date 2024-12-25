#ifndef SPARSEMAT_H_INCLUDED
#define SPARSEMAT_H_INCLUDED

#include "typedef.h"
#include "gcnLayer.h"

void aggregate(gcnLayer* layer, Matrix* X, Matrix* Y, int step, Timer *time, Stats *stats);


#endif // SPARSEMAT_H_INCLUDED
