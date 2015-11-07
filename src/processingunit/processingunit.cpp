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
	deleteAllArrays();
}

void ProcessingUnit::deleteAllArrays() {
	deleteAllDoubleArrays();
	deleteAllDoublePointerArrays();

	deleteAllIntArrays();
	deleteAllIntPonterArrays();

	deleteAllDoublePinnedArrays();
}

int ProcessingUnit::getDeviceNumber() {
	return deviceNumber;
}

double* ProcessingUnit::newDoubleArray(int size) {
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

unsigned short int* ProcessingUnit::newUnsignedShortIntArray(int size) {
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

double* ProcessingUnit::getDoublePinnedArray(int size) {
	double* array;

	cudaMallocHost ( (void**)&array, size * sizeof(double) );

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
