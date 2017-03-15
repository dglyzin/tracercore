CC=mpiCC
CFLAGS=-c -O3 -Wall

CUDACC=nvcc
CUFLAGS=-c -O3
CUDAINC=/usr/local/cuda/include
CUDAARCH=-arch=sm_20

SRC=src

SRCBLC=$(SRC)/blocks

SRCPROCUNIT=$(SRC)/processingunit
SRCPROCUNITCPU=$(SRCPROCUNIT)/cpu
SRCPROCUNITGPU=$(SRCPROCUNIT)/gpu

#SRCSTEPSTORAGE=$(SRC)/stepstorage
SRCNUMERICALMETHOD=$(SRC)/numericalmethod

SRCPROBLEM=$(SRC)/problem

SRCINTERCONNECT=$(SRC)/interconnect
SRCINTERONNECTTRANSFER=$(SRCINTERCONNECT)/transferinterconnect


BIN=bin
MPILIB18=-I/usr/mpi/gcc/openmpi-1.8.4/include -L /usr/mpi/gcc/openmpi-1.8.4/lib -lmpi -lmpi_cxx
MPILIB110=-I/usr/mpi/gcc/openmpi-1.10.6/include -L /usr/mpi/gcc/openmpi-1.10.6/lib -lmpi -lmpi_cxx
MPILIB2=-I/usr/mpi/gcc/openmpi-2.0.2/include -L /usr/mpi/gcc/openmpi-2.0.2/lib -lmpi 
MPILIB88=-I/usr/mpi/gcc/openmpi-1.8.8/include -L /usr/mpi/gcc/openmpi-1.8.8/lib -lmpi -lmpi_cxx


MPILIB=-I/usr/mpi/gcc/openmpi-1.8.4-numa/include -L /usr/mpi/gcc/openmpi-1.8.4-numa/lib -lmpi -lmpi_cxx


USERFUNCLIB=./bin -l userfuncs

BLOCK=$(SRCBLC)/block.cpp $(SRCBLC)/realblock.cpp $(SRCBLC)/nullblock.cpp

PROCUNITCPU=$(SRCPROCUNITCPU)/cpu.cpp $(SRCPROCUNITCPU)/cpu1d.cpp $(SRCPROCUNITCPU)/cpu2d.cpp $(SRCPROCUNITCPU)/cpu3d.cpp
PROCUNITGPU=$(SRCPROCUNITGPU)/gpu.cpp $(SRCPROCUNITGPU)/gpu1d.cpp $(SRCPROCUNITGPU)/gpu2d.cpp $(SRCPROCUNITGPU)/gpu3d.cpp
PROCUNIT=$(SRCPROCUNIT)/processingunit.cpp $(PROCUNITCPU) $(PROCUNITGPU)

#STEPSTORAGE=$(SRCSTEPSTORAGE)/stepstorage.cpp $(SRCSTEPSTORAGE)/eulerstorage.cpp $(SRCSTEPSTORAGE)/rk4storage.cpp $(SRCSTEPSTORAGE)/dp45storage.cpp
NUMERICALMETHOD=$(SRCNUMERICALMETHOD)/numericalmethod.cpp $(SRCNUMERICALMETHOD)/euler.cpp $(SRCNUMERICALMETHOD)/rungekutta4.cpp $(SRCNUMERICALMETHOD)/dormandprince45.cpp

#PROBLEM=$(SRCPROBLEM)/problemtype.cpp $(SRCPROBLEM)/ordinary.cpp $(SRCPROBLEM)/delay.cpp
PROBLEM=$(SRCPROBLEM)/ismartcopy.cpp $(SRCPROBLEM)/problem.cpp $(SRCPROBLEM)/ordinaryproblem.cpp

INTERCONNECT=$(SRCINTERCONNECT)/interconnect.cpp $(SRCINTERONNECTTRANSFER)/transferinterconnect.cpp $(SRCINTERCONNECT)/nontransferinterconnect.cpp $(SRCINTERONNECTTRANSFER)/transferinterconnectsend.cpp $(SRCINTERONNECTTRANSFER)/transferinterconnectrecv.cpp


SOURCE=$(SRC)/main.cpp $(SRC)/domain.cpp $(SRC)/state.cpp $(SRC)/utils.cpp $(SRC)/logger.cpp $(BLOCK) $(PROCUNIT) $(NUMERICALMETHOD) $(PROBLEM) $(INTERCONNECT)

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
	
	rm -rf $(SRCBLC)/*.o
	
	rm -rf $(SRCPROCUNIT)/*.o
	rm -rf $(SRCPROCUNITCPU)/*.o
	rm -rf $(SRCPROCUNITGPU)/*.o
	
	rm -rf $(SRCNUMERICALMETHOD)/*.o
	
	rm -rf $(SRCPROBLEM)/*.o
	
	rm -rf $(SRCINTERCONNECT)/*.o
	rm -rf $(SRCINTERONNECTTRANSFER)/*.o
	
	rm $(BIN)/$(EXECUTABLE)