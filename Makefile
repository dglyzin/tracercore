CC=mpiCC
CFLAGS=-c -O3 -Wall
SRC=src
SRCSOL=src/solvers

CUDACC=nvcc
CUFLAGS=-c -O3
CUDAINC=/usr/local/cuda/include
CUDAARCH=-arch=sm_20 
BIN=bin
MPILIB=-I/usr/mpi/gcc/openmpi-1.8.4/include -L /usr/mpi/gcc/openmpi-1.8.4/lib -lmpi -lmpi_cxx

all: HS solver

HS: main.o domain.o block.o blockcpu.o blocknull.o interconnect.o enums.o #blockgpu.o
	$(CUDACC) -O3 $(CUDAARCH) -I$(CUDAINC) $(MPILIB) -L./bin -luserfuncs $(BIN)/main.o $(BIN)/domain.o $(BIN)/block.o $(BIN)/interconnect.o  $(BIN)/blockcpu.o $(BIN)/blocknull.o $(BIN)/enums.o -o $(BIN)/HS -Xcompiler -fopenmp

#HS: main.o domain.o block.o blockcpu.o blocknull.o interconnect.o enums.o userfuncs.o #blockgpu.o
#	$(CUDACC) -O3 $(CUDAARCH) -I$(CUDAINC) $(MPILIB) $(BIN)/userfuncs.o $(BIN)/main.o $(BIN)/domain.o $(BIN)/block.o $(BIN)/interconnect.o  $(BIN)/blockcpu.o $(BIN)/blocknull.o $(BIN)/enums.o -o $(BIN)/HS -Xcompiler -fopenmp

#userfuncs.o: $(SRC)/userfuncs.cpp
#	$(CC) $(CFLAGS) $(SRC)/userfuncs.cpp -o $(BIN)/userfuncs.o
	
main.o: $(SRC)/main.cpp
	$(CC) $(CFLAGS) $(SRC)/main.cpp -o $(BIN)/main.o

domain.o: $(SRC)/domain.cpp  
	$(CC) $(CFLAGS) -I$(CUDAINC) $(SRC)/domain.cpp -fopenmp -o $(BIN)/domain.o

block.o: $(SRC)/block.cpp  
	$(CC) $(CFLAGS)  -I$(CUDAINC) $(SRC)/block.cpp -o $(BIN)/block.o
	
blockcpu.o: $(SRC)/blockcpu.cpp  
	$(CC) $(CFLAGS) -I$(CUDAINC) $(SRC)/blockcpu.cpp -o $(BIN)/blockcpu.o -fopenmp
	
#blockgpu.o: $(SRC)/blockgpu.cu  
#	$(CUDACC) $(CUFLAGS) $(CUDAARCH) -I$(CUDAINC) $(SRC)/blockgpu.cu -o $(BIN)/blockgpu.o
	
blocknull.o: $(SRC)/blocknull.cpp  
	$(CC) $(CFLAGS) $(SRC)/blocknull.cpp -o $(BIN)/blocknull.o

interconnect.o: $(SRC)/interconnect.cpp
	$(CC) $(CFLAGS) $(SRC)/interconnect.cpp -o $(BIN)/interconnect.o
	
enums.o: $(SRC)/enums.cpp
	$(CC) $(CFLAGS) $(SRC)/enums.cpp -o $(BIN)/enums.o
	

solver: solver.o
	echo "hello"

solver.o: $(SRCSOL)/solver.cpp
	$(CC) $(CFLAGS) $(SRCSOL)/solver.cpp -o $(BIN)/solver.o

clean:
	rm -rf $(BIN)/*.o $(BIN)/HS
