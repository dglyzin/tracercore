/*
 * gpu2d.cpp
 *
 *  Created on: 11 апр. 2016 г.
 *      Author: frolov
 */

#include "gpu2d.h"

GPU_2d::GPU_2d(int _deviceNumber) :
		GPU(_deviceNumber) {
}

GPU_2d::~GPU_2d() {
}

void GPU_2d::computeBorder(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result, double** source,
		double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize) {
	cudaSetDevice(deviceNumber);
	computeBorderGPU_2d();
}

void GPU_2d::computeCenter(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result, double** source,
		double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize) {
	cudaSetDevice(deviceNumber);
	computeCenterGPU_2d();
}

void GPU_2d::printArray(double* array, int zCount, int yCount, int xCount, int cellSize) {
	cudaSetDevice(deviceNumber);

	int size = zCount * yCount * xCount * cellSize;

	double* tmpArray = new double[size];

	cudaMemcpy(array, tmpArray, size * sizeof(double), cudaMemcpyDeviceToHost);

	printArray2d(array, xCount, yCount, cellSize);

	delete tmpArray;
}
