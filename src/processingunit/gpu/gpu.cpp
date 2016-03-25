/*
 * gpu.cpp
 *
 *  Created on: 25 марта 2016 г.
 *      Author: frolov
 */

#include "gpu.h"

GPU::GPU(int _deviceNumber) :
		ProcessingUnit(_deviceNumber) {
}

GPU::~GPU() {
	deleteAllArrays();
}

void GPU::prepareBorder(double* result, double* source, int zStart, int zStop, int yStart, int yStop, int xStart,
		int xStop, int yCount, int xCount, int cellSize) {
	printf("\nGPU prepare border DON'T WORK!\n");
}

void GPU::initState(double* state, initfunc_fill_ptr_t* userInitFuncs, unsigned short int* initFuncNumber,
		int blockNumber, double time) {
	//userInitFuncs[blockNumber](state, initFuncNumber);
	printf("\nGPU init state DON'T WORK! Не понятен механизм работы\n");
}

int GPU::getType() {
	return GPU_UNIT;
}

bool GPU::isCPU() {
	return false;
}

bool GPU::isGPU() {
	return true;
}

double* GPU::getDoubleArray(int size) {
	double* array;

	cudaMalloc((void**) &array, size * sizeof(double));

	return array;
}

double** GPU::getDoublePointerArray(int size) {
	double** array;

	cudaMalloc((void**) &array, size * sizeof(double*));

	return array;
}
