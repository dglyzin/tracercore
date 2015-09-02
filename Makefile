CC=mpiCC
CFLAGS=-c -O3 -Wall

CUDACC=nvcc
CUFLAGS=-c -O3
CUDAINC=/usr/local/cuda/include
CUDAARCH=-arch=sm_20

SRC=src

SRCSOL=$(SRC)/solvers
SRCSOLEULER=$(SRCSOL)/euler
SRCSOLRK4=$(SRCSOL)/rk4
SRCSOLDP45=$(SRCSOL)/dp45

SRCBLC=$(SRC)/blocks
SRCBLCCPU=$(SRCBLC)/cpu
SRCBLCGPU=$(SRCBLC)/gpu


BIN=bin
MPILIB=-I/usr/mpi/gcc/openmpi-1.8.4/include -L /usr/mpi/gcc/openmpi-1.8.4/lib -lmpi -lmpi_cxx -lmysqlcppconn

USERFUNCLIB=./bin -l userfuncs


BLOCKCPU=$(SRCBLCCPU)/blockcpu.cpp $(SRCBLCCPU)/blockcpu1d.cpp $(SRCBLCCPU)/blockcpu2d.cpp $(SRCBLCCPU)/blockcpu3d.cpp
BLOCKGPU=$(SRCBLCGPU)/blockgpu.cpp $(SRCBLCGPU)/blockgpu1d.cpp $(SRCBLCGPU)/blockgpu2d.cpp $(SRCBLCGPU)/blockgpu3d.cpp
BLOCK=$(SRCBLC)/block.cpp $(SRCBLC)/blocknull.cpp $(BLOCKCPU) $(BLOCKGPU)


SOLVEREULER=$(SRCSOLEULER)/eulersolver.cpp $(SRCSOLEULER)/eulersolvercpu.cpp $(SRCSOLEULER)/eulersolvergpu.cpp
SOLVERRK4=$(SRCSOLRK4)/rk4solver.cpp $(SRCSOLRK4)/rk4solvercpu.cpp $(SRCSOLRK4)/rk4solvergpu.cpp
SOLVERDP45=$(SRCSOLDP45)/dp45solver.cpp $(SRCSOLDP45)/dp45solvercpu.cpp $(SRCSOLDP45)/dp45solvergpu.cpp
SOLVER=$(SRCSOL)/solver.cpp $(SOLVEREULER) $(SOLVERRK4) $(SOLVERDP45)


SOURCE=$(SRC)/main.cpp $(SRC)/domain.cpp $(SRC)/interconnect.cpp $(SRC)/enums.cpp $(SRC)/dbconnector.cpp $(BLOCK) $(SOLVER)

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
	rm -rf $(SRCSOLEULER)/*.o
	rm -rf $(SRCSOLRK4)/*.o
	rm -rf $(SRCSOLDP45)/*.o
	
	rm -rf $(SRCBLC)/*.o
	rm -rf $(SRCBLCCPU)/*.o
	rm -rf $(SRCBLCGPU)/*.o
	
	rm $(BIN)/$(EXECUTABLE)