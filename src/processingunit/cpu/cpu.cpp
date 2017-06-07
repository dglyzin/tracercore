/*
 * cpu.cpp
 *
 *  Created on: 18 сент. 2015 г.
 *      Author: frolov
 */

#include "../../processingunit/cpu/cpu.h"

using namespace std;

CPU::CPU(int _deviceNumber) :
		ProcessingUnit(_deviceNumber) {
}

CPU::~CPU() {
	deleteAllArrays();
}

void CPU::prepareBorder(double* result, double* source, int zStart, int zStop, int yStart, int yStop, int xStart,
		int xStop, int yCount, int xCount, int cellSize) {
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

					result[index] = source[(zShift + yShift + xShift) * cellSize + cellShift];
					index++;
				}
			}
		}
	}
}

void CPU::getSubVolume(double* result, double* source, int zStart, int zStop, int yStart, int yStop, int xStart,
		int xStop, int yCount, int xCount, int cellSize) {
	//printf("we are in cpu::getsubvolume\n");
	prepareBorder(result, source, zStart, zStop, yStart, yStop, xStart, xStop, yCount, xCount, cellSize);
}

void CPU::setSubVolume(double* result, double* source, int zStart, int zStop, int yStart, int yStop, int xStart,
		int xStop, int yCount, int xCount, int cellSize) {
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

					result[(zShift + yShift + xShift) * cellSize + cellShift] = source[index];
					index++;
				}
			}
		}
	}
}




void CPU::initState(double* state, initfunc_fill_ptr_t* userInitFuncs, unsigned short int* initFuncNumber,
		int blockNumber, double time) {
	userInitFuncs[blockNumber](state, initFuncNumber);
}

void CPU::delayFunction(double* state, initfunc_fill_ptr_t* userInitFuncs, unsigned short int* initFuncNumber,
		int blockNumber, double time) {
	// TODO: Реализовать функции для запаздывания. Реализовать параллельные расчеты
	userInitFuncs[blockNumber](state, initFuncNumber);
}

int CPU::getType() {
	return CPUNIT;
}

bool CPU::isCPU() {
	return true;
}

bool CPU::isGPU() {
	return false;
}

bool CPU::isDeviceType(int type) {
	if (type == CPUNIT) {
		return true;
	}

	return false;
}

double* CPU::getDoubleArray(int size) {
	return new double[size];
}

double** CPU::getDoublePointerArray(int size) {
	return new double*[size];
}

int* CPU::getIntArray(int size) {
	return new int[size];
}

int** CPU::getIntPointerArray(int size) {
	return new int*[size];
}

unsigned short int* CPU::getUnsignedShortIntArray(int size) {
	return new unsigned short int[size];
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

void CPU::deallocDeviceSpecificArray(unsigned short int* toDelete) {
	delete toDelete;
}

void CPU::swapArray(double** firstArray, double** secondArray) {
	double* tmp = *firstArray;
	*firstArray = *secondArray;
	*secondArray = tmp;
}

void CPU::copyArray(double* source, double* destination, int size) {
	for (int i = 0; i < size; ++i) {
		destination[i] = source[i];
	}
}

void CPU::copyArray(unsigned short int* source, unsigned short int* destination, int size) {
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

double CPU::sumArrayElements(double* arg, int size) {
	double sum = 0;
#pragma omp parallel for reduction (+:sum)
	for (int i = 0; i < size; ++i) {
		sum += arg[i];
	}

	return sum;
}

void CPU::maxElementsElementwise(double* result, double* arg1, double* arg2, int size) {
#pragma omp parallel for
	for (int i = 0; i < size; ++i) {
		result[i] = max(arg1[i], arg2[i]);
	}
}

void CPU::maxAbsElementsElementwise(double* result, double* arg1, double* arg2, int size) {
#pragma omp parallel for
	for (int i = 0; i < size; ++i) {
		result[i] = max(abs(arg1[i]), abs(arg2[i]));
	}
}

void CPU::divisionArraysElementwise(double* result, double* arg1, double* arg2, int size) {
#pragma omp parallel for
	for (int i = 0; i < size; ++i) {
		result[i] = arg1[i] / arg2[i];
	}
}

void CPU::addNumberToArray(double* result, double* arg, double number, int size) {
#pragma omp parallel for
	for (int i = 0; i < size; ++i) {
		result[i] = arg[i] + number;
	}
}

void CPU::multiplyArraysElementwise(double* result, double* arg1, double* arg2, int size) {
#pragma omp parallel for
	for (int i = 0; i < size; ++i) {
		result[i] = arg1[i] * arg2[i];
	}
}

void CPU::insertValueIntoPonterArray(double** array, double* value, int index) {
	array[index] = value;
}

bool CPU::isNan(double* array, int size) {
	/*for (int i = 0; i < size; ++i) {
	 if (isnan(array[i]))
	 return true;
	 }

	 return false;*/
	//TODO какой вариант реализации более правильный?
	bool isN = false;
#pragma omp parallel for reduction (|: isN)
	for (int i = 0; i < size; ++i) {
		isN |= isnan(array[i]);
	}

	return isN;
}

void CPU::writeArray(double* array, int size, ofstream& out) {
	out.write((char*) array, SIZE_DOUBLE * size);
}

void CPU::readArray(double* array, int size, ifstream& in) {
	in.read((char*) array, SIZE_DOUBLE * size);
}
