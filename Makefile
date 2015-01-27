all:
	mpiCC -O3 -fopenmp src/main.cpp src/Block.cpp src/Interconnect.cpp src/BlockCpu.cpp src/BlockNull.cpp src/Domain.cpp -o HS
