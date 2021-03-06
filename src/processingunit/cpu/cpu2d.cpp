/*
 * cpu2d.cpp
 *
 *  Created on: 23 сент. 2015 г.
 *      Author: frolov
 */

#include "../../processingunit/cpu/cpu2d.h"

CPU_2d::CPU_2d(int _deviceNumber) :
		CPU(_deviceNumber) {
}

CPU_2d::~CPU_2d() {
}

void CPU_2d::computeBorder(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result, double** source,
		double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize) {
# pragma omp parallel
	{
# pragma omp for
		for (int x = 0; x < xCount; ++x) {
			int xShift = x;
			for (int y = 0; y < haloSize; ++y) {
				unsigned long long yShift = xCount * y;
				//cout << "Calc y_" << y << " x_" << x << endl;
				//printf("Calc y = %d, x = %d\n", y, x);
				mUserFuncs[mCompFuncNumber[yShift + xShift]](result, source, time, x, y, 0, parametrs, externalBorder);
			}
		}

# pragma omp for
		for (int x = 0; x < xCount; ++x) {
			int xShift = x;
			for (int y = yCount - haloSize; y < yCount; ++y) {
				unsigned long long yShift = xCount * y;
				//cout << "Calc y_" << y << " x_" << x << endl;
				//printf("Calc y = %d, x = %d\n", y, x);
				mUserFuncs[mCompFuncNumber[yShift + xShift]](result, source, time, x, y, 0, parametrs, externalBorder);
			}
		}

# pragma omp for
		for (int y = haloSize; y < yCount - haloSize; ++y) {
			unsigned long long yShift = xCount * y;
			for (int x = 0; x < haloSize; ++x) {
				int xShift = x;
				//cout << "Calc y_" << y << " x_" << x << endl;
				//printf("Calc y = %d, x = %d\n", y, x);
				mUserFuncs[mCompFuncNumber[yShift + xShift]](result, source, time, x, y, 0, parametrs, externalBorder);
			}
		}

# pragma omp for
		for (int y = haloSize; y < yCount - haloSize; ++y) {
			unsigned long long yShift = xCount * y;
			for (int x = xCount - haloSize; x < xCount; ++x) {
				int xShift = x;
				//cout << "Calc y_" << y << " x_" << x << endl;
				//printf("Calc y = %d, x = %d\n", y, x);
				mUserFuncs[mCompFuncNumber[yShift + xShift]](result, source, time, x, y, 0, parametrs, externalBorder);
			}
		}
	}
}

void CPU_2d::computeCenter(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result, double** source,
		double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize) {
# pragma omp parallel
	{
# pragma omp for
		for (int y = haloSize; y < yCount - haloSize; ++y) {
			unsigned long long yShift = xCount * y;
			for (int x = haloSize; x < xCount - haloSize; ++x) {
				int xShift = x;
				//cout << "Calc y_" << y << " x_" << x << endl;
				mUserFuncs[mCompFuncNumber[yShift + xShift]](result, source, time, x, y, 0, parametrs, externalBorder);
			}
		}
	}
}

void CPU_2d::printArray(double* array, int zCount, int yCount, int xCount, int cellSize) {
	printArray2d(array, yCount, xCount, cellSize);
}
