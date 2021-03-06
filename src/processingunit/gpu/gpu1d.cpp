/*
 * gpu1d.cpp
 *
 *  Created on: 05 апр. 2016 г.
 *      Author: frolov
 */

#include "gpu1d.h"

GPU_1d::GPU_1d(int _deviceNumber) :
		GPU(_deviceNumber) {
}

GPU_1d::~GPU_1d() {
}

void GPU_1d::computeBorder(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result, double** source,
		double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize) {
	cudaSetDevice(deviceNumber);
/*# pragma omp parallel
	{
# pragma omp for
		for (int x = 0; x < haloSize; ++x) {
			//cout << "Border Calc x_" << x << endl;
			mUserFuncs[mCompFuncNumber[x]](result, source, time, x, 0, 0, parametrs, externalBorder);
		}

# pragma omp for
		for (int x = xCount - haloSize; x < xCount; ++x) {
			//cout << "Border Calc x_" << x << endl;
			mUserFuncs[mCompFuncNumber[x]](result, source, time, x, 0, 0, parametrs, externalBorder);
		}
	}*/

	//scanf("%c", &c);

	computeBorderGPU_1d();
}

void GPU_1d::computeCenter(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result, double** source,
		double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize) {
	cudaSetDevice(deviceNumber);
/*# pragma omp parallel
	{
# pragma omp for
		for (int x = haloSize; x < xCount - haloSize; ++x) {
			//cout << "Calc x_" << x << endl;
			mUserFuncs[mCompFuncNumber[x]](result, source, time, x, 0, 0, parametrs, externalBorder);
		}
	}*/

	computeCenterGPU_1d();
}

void GPU_1d::printArray(double* array, int zCount, int yCount, int xCount, int cellSize) {
	cudaSetDevice(deviceNumber);

	int size = zCount * yCount * xCount * cellSize;

	double* tmpArray = new double [size];

	cudaMemcpy(array, tmpArray, size * sizeof(double), cudaMemcpyDeviceToHost);

	printArray1d(array, xCount, cellSize);

	delete tmpArray;
}
