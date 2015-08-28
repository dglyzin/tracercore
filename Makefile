CC=mpiCC
CFLAGS=-c -O3 -Wall

CUDACC=nvcc
CUFLAGS=-c -O3
CUDAINC=/usr/local/cuda/include
CUDAARCH=-arch=sm_20

SRC=src
SRCSOL=$(SRC)/solvers
SRCBLC=$(SRC)/blocks

BIN=bin
MPILIB=-I/usr/mpi/gcc/openmpi-1.8.4/include -L /usr/mpi/gcc/openmpi-1.8.4/lib -lmpi -lmpi_cxx

USERFUNCLIB=./bin -l userfuncs



BLOCK=$(SRCBLC)/block.cpp $(SRCBLC)/blockcpu.cpp $(SRCBLC)/blocknull.cpp $(SRCBLC)/blockgpu.cpp #$(SRCBLC)/blockcpu1d.cpp $(SRCBLC)/blockcpu2d.cpp $(SRCBLC)/blockcpu3d.cpp

SOLVER=$(SRCSOL)/solver.cpp $(SRCSOL)/eulersolver.cpp $(SRCSOL)/rk4solver.cpp $(SRCSOL)/dp45solver.cpp $(SRCSOL)/eulersolvercpu.cpp $(SRCSOL)/rk4solvercpu.cpp $(SRCSOL)/dp45solvercpu.cpp $(SRCSOL)/eulersolvergpu.cpp $(SRCSOL)/rk4solvergpu.cpp $(SRCSOL)/dp45solvergpu.cpp

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
	rm -rf $(SRCBLC)/*.o
	rm $(BIN)/$(EXECUTABLE)