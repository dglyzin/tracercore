/*
 * processingunit.cpp
 *
 *  Created on: 18 сент. 2015 г.
 *      Author: frolov
 */

#include "processingunit.h"

ProcessingUnit::ProcessingUnit() {
	// TODO Auto-generated constructor stub

}

ProcessingUnit::~ProcessingUnit() {
	// TODO Auto-generated destructor stub
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
