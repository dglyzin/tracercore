CC=mpiCC
CFLAGS=-c -O3 -Wall

CUDACC=nvcc
CUFLAGS=-c -O3
CUDAINC=/usr/local/cuda/include
CUDAARCH=-arch=sm_20

SRC=src
SRCSOL=$(SRC)/solvers

BIN=bin
MPILIB=-I/usr/mpi/gcc/openmpi-1.8.4/include -L /usr/mpi/gcc/openmpi-1.8.4/lib -lmpi -lmpi_cxx

USERFUNCLIB=./bin -l userfuncs



BLOCKCPP=block.cpp blockcpu.cpp blocknull.cpp
BLOCKGPU=blockgpu.cu

SOLVER=$(SRCSOL)/solver.cpp $(SRCSOL)/eulersolver.cpp $(SRCSOL)/rk4solver.cpp $(SRCSOL)/dp45solver.cpp
SOLVERCPU=$(SRCSOL)/eulersolvercpu.cpp $(SRCSOL)/rk4solvercpu.cpp $(SRCSOL)/dp45solvercpu.cpp
SOLVERGPU=

SOURCECPP=main.cpp domain.cpp interconnect.cpp $(BLOCKCPP) $(SOLVER) $(SOLVERCPU)
SOURCECU=$(BLOCKGPU) $(SOLVERGPU)
SOURCE=$(SOURCECPP) $(SOURCECU)

OBJECTCPP=$(SOURCECPP:.cpp=.o)
OBJECTCU=$(SOURCECU:.cu=.o)
OBJECT=$(OBJECTCPP) $(OBJECTCU)

EXECUTABLE=HS

all:
	$(SOURCE)
#all: $(SOURCE) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECT)
	$(CUDACC) -O3 $(CUDAARCH) -I$(CUDAINC) $(MPILIB) -L$(USERFUNCLIB) $(OBJECTCPP) $(OBJECTCU) -o $@
	
.cpp.o:
	$(CC) $(CFLAGS) -I$(CUDAINC) -fopenmp $< -o $@
	
.cu.o:
	$(CUDACC) $(CUFLAGS) $(CUDAARCH) -I$(CUDAINC) $< -o $@
