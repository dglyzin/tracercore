all:
	mpiCC src/main.cpp src/Block.cpp src/Interconnect.cpp src/BlockCpu.cpp -o HS
