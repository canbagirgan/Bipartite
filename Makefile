CC=mpicc

CFLAGS=-Wall -O3 -Wno-unused-variable -Wno-unused-function -Wno-uninitialized -DIRECV -DISEND
LIBS=
SRC_FILE= src/typedef.c src/basic2.c src/fileio.c
MAT_FILE= Matrix/Matrix.c Matrix/sparseMat2.c
NN_FILE = NeuralNet/activationLayer.c NeuralNet/gcnLayer2.c NeuralNet/neuralNet.c NeuralNet/lossFunctions.c
OBJ_FILE= typedef.o basic2.o fileio.o Matrix.o sparseMat2.o activationLayer.o gcnLayer2.o neuralNet.o lossFunctions.o

all: typedef matrix_cmp neuralNet_cmp
	$(CC) $(CFLAGS) -o bin/gcn main.c $(OBJ_FILE) $(LIBS)
	make clean
	
typedef: build $(SRC_FILE)
	$(CC) $(CFLAGS) -c $(SRC_FILE) $(LIBS)

matrix_cmp: $(MAT_FILE)
	$(CC) $(CFLAGS) -c $(MAT_FILE) $(LIBS)

neuralNet_cmp: $(NN_FILE)
	$(CC) $(CFLAGS) -c $(NN_FILE) $(LIBS)

build:
	test -d bin || mkdir bin

.PHONY: clean
clean:
	rm -f *.o
