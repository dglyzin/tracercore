CC=mpiCC
CFLAGS=-c -O3 -Wall  
SRC=src

CUDACC=nvcc
CUFLAGS=-c -O3
CUDAINC=/usr/local/cuda/include
CUDAARCH2=
CUDAARCH=-arch=sm_20 
BIN=bin

all: HS 

HS: main.o domain.o block.o blockcpu.o blocknull.o interconnect.o enums.o blockgpu.o
	$(CUDACC) -O3 -I/usr/mpi/gcc/openmpi-1.8.4/include -L /usr/mpi/gcc/openmpi-1.8.4/lib -lmpi -lmpi_cxx  $(BIN)/main.o $(BIN)/domain.o $(BIN)/block.o $(BIN)/interconnect.o $(BIN)/blockgpu.o $(BIN)/blockcpu.o $(BIN)/blocknull.o $(BIN)/enums.o -o $(BIN)/HS -Xcompiler -fopenmp -pg
	
main.o: $(SRC)/main.cpp
	$(CC) $(CFLAGS) $(SRC)/main.cpp -o $(BIN)/main.o -pg

domain.o: $(SRC)/domain.cpp  
	$(CC) $(CFLAGS) -I$(CUDAINC) $(SRC)/domain.cpp  -fopenmp -o $(BIN)/domain.o -pg

block.o: $(SRC)/block.cpp  
	$(CC) $(CFLAGS) $(SRC)/block.cpp  -fopenmp -o $(BIN)/block.o -pg

interconnect.o: $(SRC)/interconnect.cu  
	$(CUDACC) $(CUFLAGS) $(CUDAARCH) -I$(CUDAINC) -I/usr/mpi/gcc/openmpi-1.8.4/include -L /usr/mpi/gcc/openmpi-1.8.4/lib -lmpi -lmpi_cxx $(SRC)/interconnect.cu -o $(BIN)/interconnect.o -Xcompiler -pg
	
blockcpu.o: $(SRC)/blockcpu.cpp  
	$(CC) $(CFLAGS) $(SRC)/blockcpu.cpp  -fopenmp -o $(BIN)/blockcpu.o -pg
	
blocknull.o: $(SRC)/blocknull.cpp  
	$(CC) $(CFLAGS) $(SRC)/blocknull.cpp -o $(BIN)/blocknull.o -pg
	
enums.o: $(SRC)/enums.cpp
	$(CC) $(CFLAGS) $(SRC)/enums.cpp -o $(BIN)/enums.o -pg
	
blockgpu.o: $(SRC)/blockgpu.cu  
	$(CUDACC) $(CUFLAGS) $(CUDAARCH) -I$(CUDAINC) $(SRC)/blockgpu.cu -o $(BIN)/blockgpu.o -Xcompiler -pg

clean:
	rm -rf $(BIN)/*.o $(BIN)/HS