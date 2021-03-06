/*
 * cpu3d.cpp
 *
 *  Created on: 23 сент. 2015 г.
 *      Author: frolov
 */

#include "../../processingunit/cpu/cpu3d.h"

CPU_3d::CPU_3d(int _deviceNumber) :
		CPU(_deviceNumber) {
}

CPU_3d::~CPU_3d() {
}

void CPU_3d::computeBorder(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result, double** source,
		double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize) {
# pragma omp parallel
	{
		for (int z = 0; z < haloSize; ++z) {
			unsigned long long zShift = yCount * xCount * z;
# pragma omp for
			for (int y = 0; y < yCount; ++y) {
				unsigned long long yShift = xCount * y;
				for (int x = 0; x < xCount; ++x) {
					int xShift = x;
					//cout << "Border Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[mCompFuncNumber[zShift + yShift + xShift]](result, source, time, x, y, z, parametrs,
							externalBorder);
				}
			}
		}

		for (int z = zCount - haloSize; z < zCount; ++z) {
			unsigned long long zShift = yCount * xCount * z;
# pragma omp for
			for (int y = 0; y < yCount; ++y) {
				unsigned long long yShift = xCount * y;
				for (int x = 0; x < xCount; ++x) {
					int xShift = x;
					//cout << "Border Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[mCompFuncNumber[zShift + yShift + xShift]](result, source, time, x, y, z, parametrs,
							externalBorder);
				}
			}
		}

# pragma omp for
		for (int z = haloSize; z < zCount - haloSize; ++z) {
			unsigned long long zShift = yCount * xCount * z;
			for (int y = 0; y < haloSize; ++y) {
				unsigned long long yShift = xCount * y;
				for (int x = 0; x < xCount; ++x) {
					int xShift = x;
					//cout << "Border Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[mCompFuncNumber[zShift + yShift + xShift]](result, source, time, x, y, z, parametrs,
							externalBorder);
				}
			}
		}

# pragma omp for
		for (int z = haloSize; z < zCount - haloSize; ++z) {
			unsigned long long zShift = yCount * xCount * z;
			for (int y = yCount - haloSize; y < yCount; ++y) {
				unsigned long long yShift = xCount * y;
				for (int x = 0; x < xCount; ++x) {
					int xShift = x;
					//cout << "Border Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[mCompFuncNumber[zShift + yShift + xShift]](result, source, time, x, y, z, parametrs,
							externalBorder);
				}
			}
		}

# pragma omp for
		for (int z = haloSize; z < zCount - haloSize; ++z) {
			unsigned long long zShift = yCount * xCount * z;
			for (int y = haloSize; y < yCount - haloSize; ++y) {
				unsigned long long yShift = xCount * y;
				for (int x = 0; x < haloSize; ++x) {
					int xShift = x;
					//cout << "Border Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[mCompFuncNumber[zShift + yShift + xShift]](result, source, time, x, y, z, parametrs,
							externalBorder);
				}
			}
		}

# pragma omp for
		for (int z = haloSize; z < zCount - haloSize; ++z) {
			unsigned long long zShift = yCount * xCount * z;
			for (int y = haloSize; y < yCount - haloSize; ++y) {
				unsigned long long yShift = xCount * y;
				for (int x = xCount - haloSize; x < xCount; ++x) {
					int xShift = x;
					//cout << "Border Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[mCompFuncNumber[zShift + yShift + xShift]](result, source, time, x, y, z, parametrs,
							externalBorder);
				}
			}
		}
	}
}

void CPU_3d::computeCenter(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result, double** source,
		double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize) {
# pragma omp parallel
	{
# pragma omp for
		for (int z = haloSize; z < zCount - haloSize; ++z) {
			unsigned long long zShift = yCount * xCount * z;
			for (int y = haloSize; y < yCount - haloSize; ++y) {
				unsigned long long yShift = xCount * y;
				for (int x = haloSize; x < xCount - haloSize; ++x) {
					int xShift = x;
					//cout << "Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[mCompFuncNumber[zShift + yShift + xShift]](result, source, time, x, y, z, parametrs,
							externalBorder);
				}
			}
		}
	}
}

void CPU_3d::printArray(double* array, int zCount, int yCount, int xCount, int cellSize) {
	printArray3d(array, zCount, yCount, xCount, cellSize);
}
