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

void multipliedArrayByNumber(double* array, double value, double* result, int arrayLength);

void multipliedByNumberAndSumArrays(double* array1, double value1, double* array2, double value2, double* result, int arrayLength);
void multipliedByNumberAndSumArrays(double* array1, double value1, double* array2, double value2, double* array3, double value3, double* result, int arrayLength);
void multipliedByNumberAndSumArrays(double* array1, double value1, double* array2, double value2, double* array3, double value3, double* array4, double value4, double* result, int arrayLength);
void multipliedByNumberAndSumArrays(double* array1, double value1, double* array2, double value2, double* array3, double value3, double* array4, double value4, double* array5, double value5, double* result, int arrayLength);

double sumElementOfArray(double* array, int arrayLength);

void prepareBorder();
void computeCenter();
void computeBorder();


#endif
