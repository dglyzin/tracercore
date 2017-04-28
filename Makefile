CC=mpiCC
CFLAGS=-c -O3 -Wall

CUDACC=nvcc
CUFLAGS=-c -O3
CUDAINC=/usr/local/cuda/include
CUDAARCH=-arch=sm_20

SRC=src
BIN=bin

MPILIB=-I /usr/mpi/gcc/openmpi-1.8.4/include -L /usr/mpi/gcc/openmpi-1.8.4/lib -lmpi -lmpi_cxx

USERFUNCLIB=$(BIN) -l userfuncs

SOURCE = $(shell find . -path $(SRC)/stepstorage -prune -o -name "*.cpp")
OBJECT=$(SOURCE:.cpp=.o)

EXECUTABLE=HS


all: $(EXECUTABLE)
 
$(EXECUTABLE): $(OBJECT) cuda_func.o
	$(CUDACC) -O3 $(CUDAARCH) $(MPILIB) -L $(USERFUNCLIB) $(addprefix $(BIN)/,$(notdir $(OBJECT))) $(BIN)/cuda_func.o -o $(BIN)/$(EXECUTABLE) -Xcompiler -fopenmp

.cpp.o:
	$(CC) $(CFLAGS) -I $(CUDAINC) -fopenmp $< -o $(addprefix $(BIN)/,$(notdir $@))

cuda_func.o:
	$(CUDACC) $(CUFLAGS) $(CUDAARCH) $(SRC)/cuda_func.cu -o $(BIN)/cuda_func.o

clean: 
	rm $(BIN)/*.o
	rm $(BIN)/$(EXECUTABLE)