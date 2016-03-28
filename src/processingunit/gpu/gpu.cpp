/*
 * gpu.cpp
 *
 *  Created on: 25 марта 2016 г.
 *      Author: frolov
 */

#include "gpu.h"

GPU::GPU(int _deviceNumber) :
		ProcessingUnit(_deviceNumber) {
}

GPU::~GPU() {
	deleteAllArrays();
}

void GPU::prepareBorder(double* result, double* source, int zStart, int zStop, int yStart, int yStop, int xStart,
		int xStop, int yCount, int xCount, int cellSize) {
	printf("\nGPU prepare border DON'T WORK!\n");
}

void GPU::initState(double* state, initfunc_fill_ptr_t* userInitFuncs, unsigned short int* initFuncNumber,
		int blockNumber, double time) {
	//userInitFuncs[blockNumber](state, initFuncNumber);
	printf("\nGPU init state DON'T WORK! Не понятен механизм работы\n");
}

int GPU::getType() {
	return GPU_UNIT;
}

bool GPU::isCPU() {
	return false;
}

bool GPU::isGPU() {
	return true;
}

void GPU::copyArray(double* source, double* destination, int size) {
	copyArrayGPU(source, destination, size);
}

void GPU::copyArray(unsigned short int* source, unsigned short int* destination, int size) {
	copyArrayGPU(source, destination, size);
}

void GPU::sumArrays(double* result, double* arg1, double* arg2, int size) {
	sumArraysGPU(result, arg1, arg2, size);
}

void GPU::multiplyArrayByNumber(double* result, double* arg, double factor, int size) {
	multiplyArrayByNumberGPU(result, arg, factor, size);
}

void GPU::multiplyArrayByNumberAndSum(double* result, double* arg1, double factor, double* arg2, int size) {
	multiplyArrayByNumberAndSum(result, arg1, factor, arg2, size);
}

double GPU::sumArrayElements(double* arg, int size) {
	return sumArrayElementsGPU(arg, size);
}

void GPU::maxElementsElementwise(double* result, double* arg1, double* arg2, int size) {
	maxElementsElementwiseGPU(result, arg1, arg2, size);
}

void GPU::divisionArraysElementwise(double* result, double* arg1, double* arg2, int size) {
	divisionArraysElementwiseGPU(result, arg1, arg2, size);
}

void GPU::addNumberToArray(double* result, double* arg, double number, int size) {
	addNumberToArrayGPU(result, arg, number, size);
}

void GPU::multiplyArraysElementwise(double* result, double* arg1, double* arg2, int size) {
	multiplyArraysElementwiseGPU(result, arg1, arg2, size);
}

double* GPU::getDoubleArray(int size) {
	double* array;

	cudaMalloc((void**) &array, size * sizeof(double));

	return array;
}

double** GPU::getDoublePointerArray(int size) {
	double** array;

	cudaMalloc((void**) &array, size * sizeof(double*));

	return array;
}

int* GPU::getIntArray(int size) {
	int* array;

	cudaMalloc((void**) &array, size * sizeof(int));

	return array;
}

int** GPU::getIntPointerArray(int size) {
	int** array;

	cudaMalloc((void**) &array, size * sizeof(int*));

	return array;
}

unsigned short int* GPU::getUnsignedShortIntArray(int size) {
	unsigned short int* array;

	cudaMalloc((void**) &array, size * sizeof(unsigned short int));

	return array;
}

void GPU::deallocDeviceSpecificArray(double* toDelete) {
	cudaFree(toDelete);
}

void GPU::deallocDeviceSpecificArray(double** toDelete) {
	cudaFree(toDelete);
}

void GPU::deallocDeviceSpecificArray(int* toDelete) {
	cudaFree(toDelete);
}

void GPU::deallocDeviceSpecificArray(int** toDelete) {
	cudaFree(toDelete);
}

void GPU::deallocDeviceSpecificArray(unsigned short int* toDelete) {
	cudaFree(toDelete);
}
