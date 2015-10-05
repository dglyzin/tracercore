/*
 * cpu.cpp
 *
 *  Created on: 18 сент. 2015 г.
 *      Author: frolov
 */

#include "../../processingunit/cpu/cpu.h"

CPU::CPU() {
	// TODO Auto-generated constructor stub

}

CPU::~CPU() {
	// TODO Auto-generated destructor stub
}

void CPU::prepareBorder(double* result, double* source, int zStart, int zStop, int yStart, int yStop, int xStart, int xStop, int yCount, int xCount, int cellSize) {
	int index = 0;
	for (int z = zStart; z < zStop; ++z) {
		int zShift = xCount * yCount * z;

		for (int y = yStart; y < yStop; ++y) {
			int yShift = xCount * y;

			for (int x = xStart; x < xStop; ++x) {
				int xShift = x;

				for (int c = 0; c < cellSize; ++c) {
					int cellShift = c;
					//printf("block %d is preparing border %d, x=%d, y=%d, z=%d, index=%d\n", blockNumber, borderNumber, x,y,z, index);

					result[index] = source[ (zShift + yShift + xShift)*cellSize + cellShift ];
					index++;
				}
			}
		}
	}
}

double* CPU::getDoubleArray(int size) {
	return new double [size];
}

double** CPU::getDoublePointerArray(int size) {
	return new double* [size];
}

int* CPU::getIntArray(int size) {
	return new int [size];
}

int** CPU::getIntPointerArray(int size) {
	return new int* [size];
}

void CPU::deallocDeviceSpecificArray(double* toDelete) {
	delete toDelete;
}

void CPU::deallocDeviceSpecificArray(double** toDelete) {
	delete toDelete;
}

void CPU::deallocDeviceSpecificArray(int* toDelete) {
	delete toDelete;
}

void CPU::deallocDeviceSpecificArray(int** toDelete) {
	delete toDelete;
}

void CPU::copyArray(double* source, double* destination, int size) {
	for (int i = 0; i < size; ++i) {
		destination[i] = source[i];
	}
}

void CPU::sumArrays(double* result, double* arg1, double* arg2, int size) {
#pragma omp parallel for
	for (int i = 0; i < size; ++i) {
		result[i] = arg1[i] + arg2[i];
	}
}

void CPU::multiplyArrayByNumber(double* result, double* arg, double factor, int size) {
#pragma omp parallel for
	for (int i = 0; i < size; ++i) {
		result[i] = factor * arg[i];
	}
}

void CPU::multiplyArrayByNumberAndSum(double* result, double* arg1, double factor, double* arg2, int size) {
#pragma omp parallel for
	for (int i = 0; i < size; ++i) {
		result[i] = factor * arg1[i] + arg2[i];
	}
}
