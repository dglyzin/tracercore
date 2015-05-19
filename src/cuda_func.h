#ifndef CUDA_FUNC_H
#define CUDA_FUNC_H

#define BLOCK_LENGHT_SIZE 32
#define BLOCK_WIDTH_SIZE 16

#define BLOCK_SIZE 512

#include <stdio.h>

#include "enums.h"

void assignArray(int* array, int value, int arrayLength);
void assignArray(double* array, double value, int arrayLength);

void copyArray(int* dest, int* source, int arrayLength);
void copyArray(double* dest, double* source, int arrayLength);

void sumArray(double* arg1, double* arg2, double* result, int arrayLength);

void multiplyArrayByNumber(double* array, double value, double* result,
		int arrayLength);

void multiplyByNumberAndSumArrays(double* array1, double value1, double* array2,
		double value2, double* result, int arrayLength);
void multiplyByNumberAndSumArrays(double* array1, double value1, double* array2,
		double value2, double* array3, double value3, double* result,
		int arrayLength);
void multiplyByNumberAndSumArrays(double* array1, double value1, double* array2,
		double value2, double* array3, double value3, double* array4,
		double value4, double* result, int arrayLength);
void multiplyByNumberAndSumArrays(double* array1, double value1, double* array2,
		double value2, double* array3, double value3, double* array4,
		double value4, double* array5, double value5, double* result,
		int arrayLength);
void multiplyByNumberAndSumArrays(double* array1, double value1, double* array2,
		double value2, double* array3, double value3, double* array4,
		double value4, double* array5, double value5, double* array6,
		double value6, double* result, int arrayLength);

double sumElementOfArray(double* array, int arrayLength);

double getStepErrorDP45(double* mTempStore1, double e1, double* mTempStore3,
		double e3, double* mTempStore4, double e4, double* mTempStore5,
		double e5, double* mTempStore6, double e6, double* mTempStore7,
		double e7, double* mState, double* mArg, double timeStep, double aTol,
		double rTol, double mCount);

void prepareBorder(double* dest, double* source, int borderNumber, int zCount, int yCount, int xCount);
void computeCenter();
void computeBorder();

#endif
