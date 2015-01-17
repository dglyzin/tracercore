CC=g++
CFLAGS=-c -O3 -Wall  
SRC=src
CORE=src/core

CUDACC=nvcc
CUFLAGS=-c -O3
CUDAINC=/usr/local/cuda/include
CUDAARCH2=
CUDAARCH=-arch=compute_20 
BIN=bin

all: pfrost 

pfrost: main.o domain3d.o block3d.o block3dcpu.o approx.o block3dcuda.o interconnect.o 
	$(CUDACC) $(BIN)/main.o $(BIN)/domain3d.o $(BIN)/block3d.o $(BIN)/interconnect.o $(BIN)/block3dcpu.o $(BIN)/approx.o $(BIN)/block3dcuda.o -o $(BIN)/pfrost -Xcompiler -fopenmp
	
main.o: $(SRC)/main.cpp
	$(CC) $(CFLAGS) $(SRC)/main.cpp -o $(BIN)/main.o

domain3d.o: $(CORE)/domain3d.cpp  
	$(CC) $(CFLAGS) -I$(CUDAINC) $(CORE)/domain3d.cpp  -fopenmp -o $(BIN)/domain3d.o

block3d.o: $(CORE)/block3d.cpp  
	$(CC) $(CFLAGS) $(CORE)/block3d.cpp  -fopenmp -o $(BIN)/block3d.o

interconnect.o: $(CORE)/interconnect.cpp  
	$(CUDACC) $(CUFLAGS)  -I$(CUDAINC) $(CORE)/interconnect.cpp  -o $(BIN)/interconnect.o

block3dcpu.o: $(CORE)/block3dcpu.cpp  
	$(CC) $(CFLAGS) $(CORE)/block3dcpu.cpp  -fopenmp -o $(BIN)/block3dcpu.o
	
approx.o: $(CORE)/approx.cpp  
	$(CC) $(CFLAGS) $(CORE)/approx.cpp  -fopenmp -o $(BIN)/approx.o

block3dcuda.o: $(CORE)/block3dcuda.cu
	$(CUDACC) $(CUFLAGS)  -I$(CUDAINC) $(CORE)/block3dcuda.cu -o $(BIN)/block3dcuda.o $(CUDAARCH) 

clean:
	rm -rf $(BIN)/*.o $(BIN)/pfrostMC 