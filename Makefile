all:
	mpiCC -O3 src/main.cpp src/Block.cpp src/Interconnect.cpp src/BlockCpu.cpp src/BlockNull.cpp src/Domain.cpp -o HS
