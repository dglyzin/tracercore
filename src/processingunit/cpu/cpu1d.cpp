/*
 * cpu1d.cpp
 *
 *  Created on: 23 сент. 2015 г.
 *      Author: frolov
 */

#include "../../processingunit/cpu/cpu1d.h"

CPU_1d::CPU_1d(int _deviceNumber) :
		CPU(_deviceNumber) {
}

CPU_1d::~CPU_1d() {
}

void CPU_1d::computeBorder(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result, double** source,
		double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize) {
# pragma omp parallel
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
	}

	//scanf("%c", &c);
}

void CPU_1d::computeCenter(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result, double** source,
		double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize) {
# pragma omp parallel
	{
# pragma omp for
		for (int x = haloSize; x < xCount - haloSize; ++x) {
			//cout << "Calc x_" << x << endl;
			mUserFuncs[mCompFuncNumber[x]](result, source, time, x, 0, 0, parametrs, externalBorder);
		}
	}
}

void CPU_1d::printArray(double* array, int zCount, int yCount, int xCount, int cellSize) {
	printArray1d(array, xCount, cellSize);
}
