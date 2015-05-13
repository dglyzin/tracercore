CC=mpiCC
CFLAGS=-c -O3 -Wall -std=c++11

CUDACC=nvcc
CUFLAGS=-c -O3 -std=c++11
CUDAINC=/usr/local/cuda/include
CUDAARCH=-arch=sm_20

SRC=src
SRCSOL=$(SRC)/solvers

BIN=bin
MPILIB=-I/usr/mpi/gcc/openmpi-1.8.4/include -L /usr/mpi/gcc/openmpi-1.8.4/lib -lmpi -lmpi_cxx

USERFUNCLIB=./bin -l userfuncs



BLOCK=$(SRC)/block.cpp $(SRC)/blockcpu.cpp $(SRC)/blocknull.cpp $(SRC)/blockgpu.cpp

SOLVER=$(SRCSOL)/solver.cpp $(SRCSOL)/eulersolver.cpp $(SRCSOL)/rk4solver.cpp $(SRCSOL)/dp45solver.cpp $(SRCSOL)/eulersolvercpu.cpp $(SRCSOL)/rk4solvercpu.cpp $(SRCSOL)/dp45solvercpu.cpp $(SRCSOL)/eulersolvergpu.cpp

SOURCE=$(SRC)/main.cpp $(SRC)/domain.cpp $(SRC)/interconnect.cpp $(SRC)/enums.cpp $(BLOCK) $(SOLVER)

OBJECT=$(SOURCE:.cpp=.o)

EXECUTABLE=HS


all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECT) cuda_func.o
	$(CUDACC) -O3 $(CUDAARCH) $(MPILIB) -L$(USERFUNCLIB) $(OBJECT) $(SRC)/cuda_func.o -o $(BIN)/$(EXECUTABLE) -Xcompiler -fopenmp

.cpp.o:
	$(CC) $(CFLAGS) -I$(CUDAINC) -fopenmp $< -o $@

cuda_func.o:
	$(CUDACC) $(CUFLAGS) $(CUDAARCH) $(SRC)/cuda_func.cu -o $(SRC)/cuda_func.o
	
clean:
	rm -rf $(SRC)/*.o
	rm -rf $(SRCSOL)/*.o