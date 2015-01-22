all:
	mpiCC src/main.cpp src/Block.cpp src/Interconnect.cpp src/BlockCpu.cpp src/BlockNull.cpp -o HS
