CC=mpiCC
CFLAGS=-c -O3 -Wall

CUDACC=nvcc
CUFLAGS=-c -O3
CUDAINC=/usr/local/cuda/include
CUDAARCH=-arch=sm_20

SRC=src

SRCDOM=$(SRC)/domain

SRCBLC=$(SRC)/blocks

SRCPROCUNIT=$(SRC)/processingunit
SRCPROCUNITCPU=$(SRCPROCUNIT)/cpu
SRCPROCUNITGPU=$(SRCPROCUNIT)/gpu

SRCSTEPSTORAGE=$(SRC)/stepstorage

SRCPROBLEM=$(SRC)/problem

SRCINTERCONNECT=$(SRC)/interconnect
SRCINTERONNECTTRANSFER=$(SRCINTERCONNECT)/transferinterconnect


BIN=bin
MPILIB=-I/usr/mpi/gcc/openmpi-1.8.4/include -L /usr/mpi/gcc/openmpi-1.8.4/lib -lmpi -lmpi_cxx

USERFUNCLIB=./bin -l userfuncs

BLOCK=$(SRCBLC)/block.cpp $(SRCBLC)/realblock.cpp $(SRCBLC)/nullblock.cpp

PROCUNITCPU=$(SRCPROCUNITCPU)/cpu.cpp $(SRCPROCUNITCPU)/cpu1d.cpp $(SRCPROCUNITCPU)/cpu2d.cpp $(SRCPROCUNITCPU)/cpu3d.cpp
PROCUNITGPU=$(SRCPROCUNITGPU)/gpu.cpp $(SRCPROCUNITGPU)/gpu1d.cpp $(SRCPROCUNITGPU)/gpu2d.cpp $(SRCPROCUNITGPU)/gpu3d.cpp
PROCUNIT=$(SRCPROCUNIT)/processingunit.cpp $(PROCUNITCPU) $(PROCUNITGPU)

STEPSTORAGE=$(SRCSTEPSTORAGE)/stepstorage.cpp $(SRCSTEPSTORAGE)/eulerstorage.cpp $(SRCSTEPSTORAGE)/rk4storage.cpp $(SRCSTEPSTORAGE)/dp45storage.cpp

PROBLEM=$(SRCPROBLEM)/problemtype.cpp $(SRCPROBLEM)/ordinary.cpp $(SRCPROBLEM)/delay.cpp

INTERCONNECT=$(SRCINTERCONNECT)/interconnect.cpp $(SRCINTERONNECTTRANSFER)/transferinterconnect.cpp $(SRCINTERCONNECT)/nontransferinterconnect.cpp $(SRCINTERONNECTTRANSFER)/transferinterconnectsend.cpp $(SRCINTERONNECTTRANSFER)/transferinterconnectrecv.cpp


SOURCE=$(SRC)/main.cpp $(SRCDOM)/domain.cpp $(SRC)/utils.cpp $(BLOCK) $(PROCUNIT) $(STEPSTORAGE) $(PROBLEM) $(INTERCONNECT)

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
	
	rm -rf $(SRCDOM)/*.o
	
	rm -rf $(SRCBLC)/*.o
	
	rm -rf $(SRCPROCUNIT)/*.o
	rm -rf $(SRCPROCUNITCPU)/*.o
	rm -rf $(SRCPROCUNITGPU)/*.o
	
	rm -rf $(SRCSTEPSTORAGE)/*.o
	
	rm -rf $(SRCPROBLEM)/*.o
	
	rm -rf $(SRCINTERCONNECT)/*.o
	rm -rf $(SRCINTERONNECTTRANSFER)/*.o
	
	rm $(BIN)/$(EXECUTABLE)