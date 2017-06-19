/*
 * processingunit.cpp
 *
 *  Created on: 18 сент. 2015 г.
 *      Author: frolov
 */

#include "../processingunit/processingunit.h"

using namespace std;

ProcessingUnit::ProcessingUnit(int _deviceNumber) {
	deviceNumber = _deviceNumber;

	doubleArrays.clear();
	doublePointerArrays.clear();

	intArrays.clear();
	intPointerArrays.clear();

	doublePinnedArrays.clear();

	unsignedShortIntArrays.clear();
}

ProcessingUnit::~ProcessingUnit() {
}

void ProcessingUnit::deleteAllArrays() {
	deleteAllDoubleArrays();
	deleteAllDoublePointerArrays();

	deleteAllIntArrays();
	deleteAllIntPonterArrays();

	deleteAllDoublePinnedArrays();
}

bool ProcessingUnit::isDeviceNumber(int number) {
	if (number == deviceNumber) {
		return true;
	}

	return false;
}

int ProcessingUnit::getDeviceNumber() {
	return deviceNumber;
}

double* ProcessingUnit::newDoubleArray(unsigned long long size) {
	double* array;

	array = getDoubleArray(size);

	doubleArrays.push_back(array);

	return array;
}

double** ProcessingUnit::newDoublePointerArray(int size) {
	double** array;

	array = getDoublePointerArray(size);

	doublePointerArrays.push_back(array);

	return array;
}

int* ProcessingUnit::newIntArray(int size) {
	int* array;

	array = getIntArray(size);

	intArrays.push_back(array);

	return array;
}

int** ProcessingUnit::newIntPointerArray(int size) {
	int** array;

	array = getIntPointerArray(size);

	intPointerArrays.push_back(array);

	return array;
}

double* ProcessingUnit::newDoublePinnedArray(int size) {
	double* array;

	array = getDoublePinnedArray(size);

	doublePinnedArrays.push_back(array);

	return array;
}

unsigned short int* ProcessingUnit::newUnsignedShortIntArray(unsigned long long size) {
	unsigned short int* array;

	array = getUnsignedShortIntArray(size);

	unsignedShortIntArrays.push_back(array);

	return array;
}

void ProcessingUnit::deleteDeviceSpecificArray(double* toDelete) {
	doubleArrays.remove(toDelete);

	deallocDeviceSpecificArray(toDelete);
}

void ProcessingUnit::deleteDeviceSpecificArray(double** toDelete) {
	doublePointerArrays.remove(toDelete);

	deallocDeviceSpecificArray(toDelete);
}

void ProcessingUnit::deleteDeviceSpecificArray(int* toDelete) {
	intArrays.remove(toDelete);

	deallocDeviceSpecificArray(toDelete);
}

void ProcessingUnit::deleteDeviceSpecificArray(int** toDelete) {
	intPointerArrays.remove(toDelete);

	deallocDeviceSpecificArray(toDelete);
}

void ProcessingUnit::deleteDoublePinnedArray(double* toDelete) {
	doublePinnedArrays.remove(toDelete);

	deallocDoublePinnedArray(toDelete);
}

void ProcessingUnit::deleteUnsignedShortInt(unsigned short int* toDelete) {
	unsignedShortIntArrays.remove(toDelete);

	deallocDeviceSpecificArray(toDelete);
}

/*void ProcessingUnit::swapStorages(double** sourceStorages, double** destinationStorages, int sourceStorageIndex,
		int destinationStorageIndex) {
	double* tmp = sourceStorages[sourceStorageIndex];
	sourceStorages[sourceStorageIndex] = destinationStorages[destinationStorageIndex];
	destinationStorages[destinationStorageIndex] = tmp;
}*/

void ProcessingUnit::saveArray(double* array, unsigned long long size, char* path) {
	ofstream out;
	out.open(path, ios::binary | ios::app);

	int saveSize = size % INT_MAX;
	while(saveSize > 0) {
		writeArray(array, saveSize, out);
		size -= saveSize;
		saveSize = size % INT_MAX;
	}

	out.close();
}

void ProcessingUnit::loadArray(double* array, unsigned long long size, std::ifstream& in) {
	int readSize = size % INT_MAX;
	while(readSize > 0) {
		readArray(array, size, in);
		size -= readSize;
		readSize = size % INT_MAX;
	}
}

double* ProcessingUnit::getDoublePinnedArray(int size) {
	double* array;

	cudaMallocHost((void**) &array, size * sizeof(double));

	return array;
}

void ProcessingUnit::deallocDoublePinnedArray(double* toDelete) {
	cudaFreeHost(toDelete);
}

void ProcessingUnit::deleteAllDoubleArrays() {
	list<double*>::iterator i;

	for (i = doubleArrays.begin(); i != doubleArrays.end(); ++i) {
		deallocDeviceSpecificArray(*i);
	}

	doubleArrays.clear();
}

void ProcessingUnit::deleteAllDoublePointerArrays() {
	list<double**>::iterator i;

	for (i = doublePointerArrays.begin(); i != doublePointerArrays.end(); ++i) {
		deallocDeviceSpecificArray(*i);
	}

	doublePointerArrays.clear();
}

void ProcessingUnit::deleteAllIntArrays() {
	list<int*>::iterator i;

	for (i = intArrays.begin(); i != intArrays.end(); ++i) {
		deallocDeviceSpecificArray(*i);
	}

	intArrays.clear();
}

void ProcessingUnit::deleteAllIntPonterArrays() {
	list<int**>::iterator i;

	for (i = intPointerArrays.begin(); i != intPointerArrays.end(); ++i) {
		deallocDeviceSpecificArray(*i);
	}

	intPointerArrays.clear();
}

void ProcessingUnit::deleteAllDoublePinnedArrays() {
	list<double*>::iterator i;

	for (i = doublePinnedArrays.begin(); i != doublePinnedArrays.end(); ++i) {
		deallocDoublePinnedArray(*i);
	}

	doublePinnedArrays.clear();
}

void ProcessingUnit::deleteAllUnsignedShortInt() {
	list<unsigned short int*>::iterator i;

	for (i = unsignedShortIntArrays.begin(); i != unsignedShortIntArrays.end(); ++i) {
		deallocDeviceSpecificArray(*i);
	}

	unsignedShortIntArrays.clear();
}

void ProcessingUnit::printCell(double* array, int cellSize) {
	printf("(");
	printf("%5.2f", array[0]);

	for (int h = 1; h < cellSize; ++h) {
		printf(", %5.2f", array[h]);
	}
	printf(")");
}

void ProcessingUnit::printArray1d(double* array, int xCount, int cellSize) {
	printCell(array, cellSize);

	unsigned long long shift = 0;
	for (int x = 1; x < xCount; ++x) {
		printf(" ");
		shift = x * cellSize;
		printCell(array + shift, cellSize);
	}
	printf("\n");
}

void ProcessingUnit::printArray2d(double* array, int yCount, int xCount, int cellSize) {
	unsigned long long shift = 0;
	for (int y = 0; y < yCount; ++y) {
		shift = xCount * y * cellSize;
		printArray1d(array + shift, xCount, cellSize);
	}
}

void ProcessingUnit::printArray3d(double* array, int zCount, int yCount, int xCount, int cellSize) {
	unsigned long long shift = 0;
	for (int z = 0; z < zCount; ++z) {
		printf("z = %d", z);
		shift = yCount * xCount * z * cellSize;
		printArray2d(array + shift, yCount, xCount, cellSize);
		printf("\n");
	}
}
