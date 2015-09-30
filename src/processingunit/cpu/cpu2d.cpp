/*
 * cpu2d.cpp
 *
 *  Created on: 23 сент. 2015 г.
 *      Author: frolov
 */

#include "../../processingunit/cpu/cpu2d.h"

CPU_2d::CPU_2d() {
	// TODO Auto-generated constructor stub

}

CPU_2d::~CPU_2d() {
	// TODO Auto-generated destructor stub
}

void CPU_2d::computeBorder(double* result, double** source, double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize) {
# pragma omp parallel
	{
# pragma omp for
		for (int x = 0; x < xCount; ++x) {
			int xShift = x;
			for (int y = 0; y < haloSize; ++y) {
				int yShift = xCount * y;
				//cout << "Calc y_" << y << " x_" << x << endl;
				//printf("Calc y = %d, x = %d\n", y, x);
				mUserFuncs[ mCompFuncNumber[ yShift + xShift ] ](result, source[0], time, x, y, 0, parametrs, externalBorder);
			}
		}

# pragma omp for
		for (int x = 0; x < xCount; ++x) {
			int xShift = x;
			for (int y = yCount - haloSize; y < yCount; ++y) {
				int yShift = xCount * y;
				//cout << "Calc y_" << y << " x_" << x << endl;
				//printf("Calc y = %d, x = %d\n", y, x);
				mUserFuncs[ mCompFuncNumber[ yShift + xShift ] ](result, source[0], time, x, y, 0, parametrs, externalBorder);
			}
		}

# pragma omp for
		for (int y = haloSize; y < yCount - haloSize; ++y) {
			int yShift = xCount * y;
			for (int x = 0; x < haloSize; ++x) {
				int xShift = x;
				//cout << "Calc y_" << y << " x_" << x << endl;
				//printf("Calc y = %d, x = %d\n", y, x);
				mUserFuncs[ mCompFuncNumber[ yShift + xShift ] ](result, source[0], time, x, y, 0, parametrs, externalBorder);
			}
		}

# pragma omp for
		for (int y = haloSize; y < yCount - haloSize; ++y) {
			int yShift = xCount * y;
			for (int x = xCount - haloSize; x < xCount; ++x) {
				int xShift = x;
				//cout << "Calc y_" << y << " x_" << x << endl;
				//printf("Calc y = %d, x = %d\n", y, x);
				mUserFuncs[ mCompFuncNumber[ yShift + xShift ] ](result, source[0], time, x, y, 0, parametrs, externalBorder);
			}
		}
	}
}

void CPU_2d::computeCenter(double* result, double** source, double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize) {
# pragma omp parallel
	{
# pragma omp for
		for (int y = haloSize; y < yCount - haloSize; ++y) {
			int yShift = xCount * y;
			for (int x = haloSize; x < xCount - haloSize; ++x) {
				int xShift = x;
				//cout << "Calc y_" << y << " x_" << x << endl;
				mUserFuncs[ mCompFuncNumber[ yShift + xShift ] ](result, source[0], time, x, y, 0, parametrs, externalBorder);
			}
		}
	}
}
