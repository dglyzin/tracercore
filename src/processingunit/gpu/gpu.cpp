/*
 * gpu.cpp
 *
 *  Created on: 25 марта 2016 г.
 *      Author: frolov
 */

#include "gpu.h"

using namespace std;

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

void GPU::getSubVolume(double* result, double* source, int zStart, int zStop, int yStart, int yStop, int xStart,
			int xStop, int yCount, int xCount, int cellSize){
	printf("\nGPU prepare border DON'T WORK!\n");
}

void GPU::setSubVolume(double* result, double* source, int zStart, int zStop, int yStart, int yStop, int xStart,
				int xStop, int yCount, int xCount, int cellSize){
	printf("\nGPU prepare border DON'T WORK!\n");
}

void GPU::initState(double* state, initfunc_fill_ptr_t* userInitFuncs, unsigned short int* initFuncNumber,
		int blockNumber, double time) {
	//userInitFuncs[blockNumber](state, initFuncNumber);
	printf("\nGPU init state DON'T WORK! Не понятен механизм работы\n");
}

void GPU::delayFunction(double* state, initfunc_fill_ptr_t* userInitFuncs, unsigned short int* initFuncNumber,
		int blockNumber, double time) {

}

int GPU::getType() {
	return GPUNIT;
}

bool GPU::isCPU() {
	return false;
}

bool GPU::isGPU() {
	return true;
}

bool GPU::isDeviceType(int type) {
	if (type == GPUNIT) {
		return true;
	}

	return false;
}

void GPU::swapArray(double** firstArray, double** secondArray) {
	cudaSetDevice(deviceNumber);
	// TODO: swap array for GPU
}

void GPU::copyArray(double* source, double* destination, unsigned long long size) {
	cudaSetDevice(deviceNumber);
	cudaMemcpy(destination, source, size * SIZE_DOUBLE, cudaMemcpyDeviceToDevice);
}

void GPU::copyArrayHostToDevice(unsigned short int* source, unsigned short int* destination, unsigned long long size) {
	cudaSetDevice(deviceNumber);
	cudaMemcpy(destination, source, size * SIZE_UN_SH_INT, cudaMemcpyHostToDevice);
}

void GPU::sumArrays(double* result, double* arg1, double* arg2, unsigned long long size) {
	cudaSetDevice(deviceNumber);
	sumArraysGPU(result, arg1, arg2, size);
}

void GPU::multiplyArrayByNumber(double* result, double* arg, double factor, unsigned long long size) {
	cudaSetDevice(deviceNumber);
	multiplyArrayByNumberGPU(result, arg, factor, size);
}

void GPU::multiplyArrayByNumberAndSum(double* result, double* arg1, double factor, double* arg2, unsigned long long size) {
	cudaSetDevice(deviceNumber);
	multiplyArrayByNumberAndSumGPU(result, arg1, factor, arg2, size);
}

double GPU::sumArrayElements(double* arg, unsigned long long size) {
	cudaSetDevice(deviceNumber);
	return sumArrayElementsGPU(arg, size);
}

void GPU::maxElementsElementwise(double* result, double* arg1, double* arg2, unsigned long long size) {
	cudaSetDevice(deviceNumber);
	maxElementsElementwiseGPU(result, arg1, arg2, size);
}

void GPU::maxAbsElementsElementwise(double* result, double* arg1, double* arg2, unsigned long long size) {
	cudaSetDevice(deviceNumber);
	maxAbsElementsElementwiseGPU(result, arg1, arg2, size);
}

void GPU::divisionArraysElementwise(double* result, double* arg1, double* arg2, unsigned long long size) {
	cudaSetDevice(deviceNumber);
	divisionArraysElementwiseGPU(result, arg1, arg2, size);
}

void GPU::addNumberToArray(double* result, double* arg, double number, unsigned long long size) {
	cudaSetDevice(deviceNumber);
	addNumberToArrayGPU(result, arg, number, size);
}

void GPU::multiplyArraysElementwise(double* result, double* arg1, double* arg2, unsigned long long size) {
	cudaSetDevice(deviceNumber);
	multiplyArraysElementwiseGPU(result, arg1, arg2, size);
}

void GPU::insertValueIntoPonterArray(double** array, double* value, int index) {
	cudaSetDevice(deviceNumber);
	cudaMemcpy(array + index, value, 1 * sizeof(double*), cudaMemcpyHostToDevice);
}

bool GPU::isNan(double* array, unsigned long long size) {
	cudaSetDevice(deviceNumber);
	return isNanGPU(array, size);
}

double* GPU::getDoubleArray(unsigned long long size) {
	cudaSetDevice(deviceNumber);

	double* array;

	cudaMalloc((void**) &array, size * sizeof(double));

	return array;
}

double** GPU::getDoublePointerArray(int size) {
	cudaSetDevice(deviceNumber);

	double** array;

	cudaMalloc((void**) &array, size * sizeof(double*));

	return array;
}

int* GPU::getIntArray(int size) {
	cudaSetDevice(deviceNumber);

	int* array;

	cudaMalloc((void**) &array, size * sizeof(int));

	return array;
}

int** GPU::getIntPointerArray(int size) {
	cudaSetDevice(deviceNumber);

	int** array;

	cudaMalloc((void**) &array, size * sizeof(int*));

	return array;
}

unsigned short int* GPU::getUnsignedShortIntArray(unsigned long long size) {
	cudaSetDevice(deviceNumber);

	unsigned short int* array;

	cudaMalloc((void**) &array, size * sizeof(unsigned short int));

	return array;
}

void GPU::deallocDeviceSpecificArray(double* toDelete) {
	cudaSetDevice(deviceNumber);
	cudaFree(toDelete);
}

void GPU::deallocDeviceSpecificArray(double** toDelete) {
	cudaSetDevice(deviceNumber);
	cudaFree(toDelete);
}

void GPU::deallocDeviceSpecificArray(int* toDelete) {
	cudaSetDevice(deviceNumber);
	cudaFree(toDelete);
}

void GPU::deallocDeviceSpecificArray(int** toDelete) {
	cudaSetDevice(deviceNumber);
	cudaFree(toDelete);
}

void GPU::deallocDeviceSpecificArray(unsigned short int* toDelete) {
	cudaSetDevice(deviceNumber);
	cudaFree(toDelete);
}

void GPU::writeArray(double* array, int size, ofstream& out) {
	cudaSetDevice(deviceNumber);
	// TODO ПРОВЕРИТЬ КОПИРОВАНИЕ!!!!

	double* tmpArray = new double[size];

	cudaMemcpy(array, tmpArray, size * sizeof(double), cudaMemcpyDeviceToHost);

	out.write((char*) tmpArray, SIZE_DOUBLE * size);

	delete tmpArray;
}

void GPU::readArray(double* array, int size, ifstream& in) {
	cudaSetDevice(deviceNumber);

	double* tmpArray = new double[size];

	in.read((char*) tmpArray, SIZE_DOUBLE * size);

	cudaMemcpy(tmpArray, array, size * sizeof(double), cudaMemcpyHostToDevice);

	delete tmpArray;
}
