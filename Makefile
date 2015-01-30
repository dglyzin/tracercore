CC=mpiCC
CFLAGS=-c -O3 -Wall  
SRC=src

CUDACC=nvcc
CUFLAGS=-c -O3
CUDAINC=/usr/local/cuda/include
CUDAARCH2=
CUDAARCH=-arch=compute_20 
BIN=bin

all: HS 

HS: main.o Domain.o Block.o BlockCpu.o BlockNull.o Interconnect.o BlockGpu.o
	$(CUDACC) -O3 -I/usr/mpi/gcc/openmpi-1.8.4/include -L /usr/mpi/gcc/openmpi-1.8.4/lib -lmpi_cxx  $(BIN)/main.o $(BIN)/Domain.o $(BIN)/Block.o $(BIN)/Interconnect.o $(BIN)/BlockCpu.o $(BIN)/BlockNull.o $(BIN)/BlockGpu.o -o $(BIN)/HS -Xcompiler -fopenmp
	
main.o: $(SRC)/main.cpp
	$(CC) $(CFLAGS) $(SRC)/main.cpp -o $(BIN)/main.o

Domain.o: $(SRC)/Domain.cpp  
	$(CC) $(CFLAGS) -I$(CUDAINC) $(SRC)/Domain.cpp  -fopenmp -o $(BIN)/Domain.o

Block.o: $(SRC)/Block.cpp  
	$(CC) $(CFLAGS) $(SRC)/Block.cpp  -fopenmp -o $(BIN)/Block.o

Interconnect.o: $(SRC)/Interconnect.cpp  
	$(CC) $(CFLAGS) $(SRC)/Interconnect.cpp -o $(BIN)/Interconnect.o
	
BlockCpu.o: $(SRC)/BlockCpu.cpp  
	$(CC) $(CFLAGS) $(SRC)/BlockCpu.cpp  -fopenmp -o $(BIN)/BlockCpu.o
	
BlockNull.o: $(SRC)/BlockNull.cpp  
	$(CC) $(CFLAGS) $(SRC)/BlockNull.cpp -o $(BIN)/BlockNull.o
	
BlockGpu.o: $(SRC)/BlockGpu.cu  
	$(CUDACC) $(CUFLAGS)  -I$(CUDAINC) $(SRC)/BlockGpu.cu -o $(BIN)/BlockGpu.o

clean:
	rm -rf $(BIN)/*.o $(BIN)/pfrostMC 