#ifndef CUDA_FUNC_H
#define CUDA_FUNC_H

#define BLOCK_LENGHT_SIZE 32
#define BLOCK_WIDTH_SIZE 16

#define BLOCK_SIZE 512

#include <stdio.h>

#include "enums.h"

void copyArrayGPU(double* source, double* destination, int size);
void copyArrayGPU(unsigned short int* source, unsigned short int* destination, int size);

void sumArraysGPU(double* result, double* arg1, double* arg2, int size);
void multiplyArrayByNumberGPU(double* result, double* arg, double factor, int size);
void multiplyArrayByNumberAndSumGPU(double* result, double* arg1, double factor, double* arg2, int size);

double sumArrayElementsGPU(double* arg, int size);
void maxElementsElementwiseGPU(double* result, double* arg1, double* arg2, int size);
void divisionArraysElementwiseGPU(double* result, double* arg1, double* arg2, int size);

void addNumberToArrayGPU(double* result, double* arg, double number, int size);
void multiplyArraysElementwiseGPU(double* result, double* arg1, double* arg2, int size);

bool isNanGPU(double* array, int size);

void prepareBorderCudaFunc(double* source, int borderNumber, int zStart, int zStop, int yStart, int yStop, int xStart,
		int xStop, double** blockBorder, int zCount, int yCount, int xCount, int cellSize);
void computeCenter();
void computeBorder();

#endif
