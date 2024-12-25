#ifndef NEURALNET_H_INCLUDED
#define NEURALNET_H_INCLUDED

#include "activationLayer.h"
#include "gcnLayer.h"
#include "typedef.h"

enum layer_type {
    ACTIVATION,
    GCN
};

typedef struct {
    enum layer_type type;
    activationLayer* activation_layer;
    gcnLayer* gcn_layer;
} layer_super;

typedef struct {
    layer_super **layers;
    int n_layers;
    int layer_capacity;
    double lr;
} neural_net;

neural_net* net_init(int capacity);
layer_super* layer_init(enum layer_type);
layer_super* layer_init_activation(enum activation_type type);
layer_super* layer_init_gcn(SparseMat* adj, SparseMat* adj_T, int size_f, int size_out);
void net_addLayer(neural_net* net, layer_super* layer);
ParMatrix* net_forward(neural_net* net, ParMatrix* input, int option, Timer *time, Stats *stats);
void net_backward(neural_net* net, Matrix* error, double lr, int t, Timer *time, Stats *stats);
void net_free(neural_net* net);
void layer_free(layer_super* layer);

#endif // NEURALNET_H_INCLUDED
