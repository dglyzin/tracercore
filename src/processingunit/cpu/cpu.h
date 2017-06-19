/*
 * cpu.h
 *
 *  Created on: 18 сент. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_PROCESSINGUNIT_CPU_CPU_H_
#define SRC_PROCESSINGUNIT_CPU_CPU_H_

#include <omp.h>
#include <math.h>

#include <string.h>

#include "../processingunit.h"

class CPU: public ProcessingUnit {
public:
	CPU(int _deviceNumber);
	virtual ~CPU();

	void prepareBorder(double* result, double* source, int zStart, int zStop, int yStart, int yStop, int xStart,
			int xStop, int yCount, int xCount, int cellSize);

	void getSubVolume(double* result, double* source, int zStart, int zStop, int yStart, int yStop, int xStart,
			int xStop, int yCount, int xCount, int cellSize);

	void setSubVolume(double* result, double* source, int zStart, int zStop, int yStart, int yStop, int xStart,
				int xStop, int yCount, int xCount, int cellSize);

	void initState(double* state, initfunc_fill_ptr_t* userInitFuncs, unsigned short int* initFuncNumber,
			int blockNumber, double time);

	void delayFunction(double* state, initfunc_fill_ptr_t* userInitFuncs, unsigned short int* initFuncNumber,
			int blockNumber, double time);

	int getType();

	bool isCPU();
	bool isGPU();

	bool isDeviceType(int type);

	void swapArray(double** firstArray, double** secondArray);

	void copyArray(double* source, double* destination, unsigned long long size);
	void copyArray(unsigned short int* source, unsigned short int* destination, unsigned long long size);

	void sumArrays(double* result, double* arg1, double* arg2, unsigned long long size);
	void multiplyArrayByNumber(double* result, double* arg, double factor, unsigned long long size);
	void multiplyArrayByNumberAndSum(double* result, double* arg1, double factor, double* arg2, unsigned long long size);

	double sumArrayElements(double* arg, unsigned long long size);
	void maxElementsElementwise(double* result, double* arg1, double* arg2, unsigned long long size);
	void maxAbsElementsElementwise(double* result, double* arg1, double* arg2, unsigned long long size);
	void divisionArraysElementwise(double* result, double* arg1, double* arg2, unsigned long long size);

	void addNumberToArray(double* result, double* arg, double number, unsigned long long size);
	void multiplyArraysElementwise(double* result, double* arg1, double* arg2, unsigned long long size);

	void insertValueIntoPonterArray(double** array, double* value, int index);

	bool isNan(double* array, unsigned long long size);

protected:
	double* getDoubleArray(unsigned long long size);
	double** getDoublePointerArray(int size);

	int* getIntArray(int size);
	int** getIntPointerArray(int size);

	unsigned short int* getUnsignedShortIntArray(unsigned long long size);

	void deallocDeviceSpecificArray(double* toDelete);
	void deallocDeviceSpecificArray(double** toDelete);

	void deallocDeviceSpecificArray(int* toDelete);
	void deallocDeviceSpecificArray(int** toDelete);

	void deallocDeviceSpecificArray(unsigned short int* toDelete);

	void writeArray(double* array, int size, std::ofstream& out);
	void readArray(double* array, int size, std::ifstream& in);
};

#endif /* SRC_PROCESSINGUNIT_CPU_CPU_H_ */
