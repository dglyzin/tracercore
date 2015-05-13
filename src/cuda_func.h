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

void multipliedArrayByNumber(double* array, double value, int arrayLength);

void prepareArgument();
void prepareBorder();
void computeCenter();
void computeBorder();


#endif
