CC=mpiCC
CFLAGS=-c -O3 -Wall -std=c++11

CUDACC=nvcc
CUFLAGS=-c -O3
CUDAINC=/usr/local/cuda/include
CUDAARCH=-arch=sm_20

SRC=src
SRCSOL=$(SRC)/solvers

BIN=bin
MPILIB=-I/usr/mpi/gcc/openmpi-1.8.4/include -L /usr/mpi/gcc/openmpi-1.8.4/lib -lmpi -lmpi_cxx

USERFUNCLIB=./bin -l userfuncs



BLOCKCPP=$(SRC)/block.cpp $(SRC)/blockcpu.cpp $(SRC)/blocknull.cpp
BLOCKGPU=$(SRC)/blockgpu.cu

SOLVER=$(SRCSOL)/solver.cpp $(SRCSOL)/eulersolver.cpp $(SRCSOL)/rk4solver.cpp $(SRCSOL)/dp45solver.cpp
SOLVERCPU=$(SRCSOL)/eulersolvercpu.cpp $(SRCSOL)/rk4solvercpu.cpp $(SRCSOL)/dp45solvercpu.cpp
SOLVERGPU=

SOURCECPP=$(SRC)/main.cpp $(SRC)/domain.cpp $(SRC)/interconnect.cpp $(SRC)/enums.cpp $(BLOCKCPP) $(SOLVER) $(SOLVERCPU)
SOURCECU=$(BLOCKGPU) $(SOLVERGPU)
SOURCE=$(SOURCECPP) $(SOURCECU)

OBJECTCPP=$(SOURCECPP:.cpp=.o)
OBJECTCU=$(SOURCECU:.cu=.o)
OBJECT=$(OBJECTCPP) $(OBJECTCU) #block1.o

EXECUTABLE=HS

#all:
#	$(OBJECT) $(SOURCE)
all: $(SOURCE) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECT)
	$(CUDACC) $(CUFLAGS) $(CUDAARCH) -I$(CUDAINC) $(MPILIB) -L$(USERFUNCLIB) $(OBJECT) -o $@ -Xcompiler -fopenmp

.cpp.o:
	$(CC) $(CFLAGS) -I$(CUDAINC) -fopenmp $< -o $@

.cu.o:
	#$(CUDACC) $(CUFLAGS) $(CUDAARCH) -I$(CUDAINC) $< -o $@
	echo "11"
	
block1.o: $(SRC)/blockgpu.cu
	$(CUDACC) $(CUFLAGS) $(CUDAARCH) -I$(CUDAINC) $(SRC)/blockgpu.cu -o $(SRC)/block1.o

	
clean:
	rm -rf $(SRC)/*.o
	rm -rf $(SRCSOL)/*.o