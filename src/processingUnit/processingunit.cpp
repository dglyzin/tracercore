/*
 * processingunit.cpp
 *
 *  Created on: 18 сент. 2015 г.
 *      Author: frolov
 */

#include "processingunit.h"

using namespace std;

ProcessingUnit::ProcessingUnit() {
	doubleArrays.clear();
	doublePointerArrays.clear();

	intArrays.clear();
	intPointerArrays.clear();

	doublePinnedArrays.clear();
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

void ProcessingUnit::deleteAllDoubleArrays() {
	list<double*>::iterator i;

	for (i = doubleArrays.begin(); i < doubleArrays.end(); ++i) {
		deleteDoubleArray(*i);
	}

	doubleArrays.clear();
}

void ProcessingUnit::deleteAllDoublePointerArrays() {
	list<double**>::iterator i;

	for (i = doublePointerArrays.begin(); i < doublePointerArrays.end(); ++i) {
		deleteDoublePointerArray(*i);
	}

	doublePointerArrays.clear();
}

void ProcessingUnit::deleteAllIntArrays() {
	list<int*>::iterator i;

	for (i = intArrays.begin(); i < intArrays.end(); ++i) {
		deleteIntArray(*i)
	}

	intArrays.clear();
}

void ProcessingUnit::deleteAllIntPonterArrays() {
	list<int**>::iterator i;

	for (i = intPointerArrays.begin(); i < intPointerArrays.end(); ++i) {
		deleteIntPointerArray(*i)
	}

	intPointerArrays.clear();
}

void ProcessingUnit::deleteAllDoublePinnedArrays() {
	list<double*>::iterator i;

	for (i = doublePinnedArrays.begin(); i < doublePinnedArrays.end(); ++i) {
		deleteAllDoublePinnedArrays(*i)
	}

	doublePinnedArrays.clear();
}

double* ProcessingUnit::newDoublePinnedArray(int size) {
	double* array;

	array = getDoublePinnedArray(size);

	doublePinnedArrays.push_back(array);

	return array;
}

double* ProcessingUnit::getDoublePinnedArray(int size) {
	double* array;

	cudaMallocHost ( (void**)&array, size * sizeof(double) );

	return array;
}

void ProcessingUnit::deleteDoublePinnedArray(double* toDelete) {
	cudaFreeHost(toDelete);
}
