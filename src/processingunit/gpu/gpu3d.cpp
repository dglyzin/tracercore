/*
 * gpu3d.cpp
 *
 *  Created on: 11 апр. 2016 г.
 *      Author: frolov
 */

#include "gpu3d.h"

GPU_3d::GPU_3d(int _deviceNumber) :
		GPU(_deviceNumber) {
}

GPU_3d::~GPU_3d() {
}

void GPU_3d::computeBorder(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result, double** source,
		double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize) {
	computeBorderGPU_3d();
}

void GPU_3d::computeCenter(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result, double** source,
		double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize) {
	computeCenterGPU_3d();
}

void GPU_3d::printArray(double* array, int zCount, int yCount, int xCount, int cellSize) {
	int size = zCount * yCount * xCount * cellSize;

	double* tmpArray = new double[size];

	cudaMemcpy(array, tmpArray, size * sizeof(double), cudaMemcpyDeviceToHost);

	printArray3d(array, xCount, yCount, zCount, cellSize);

	delete tmpArray;
}
