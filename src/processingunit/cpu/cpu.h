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

#include "../processingunit.h"

class CPU: public ProcessingUnit {
public:
	CPU(int _deviceNumber);
	virtual ~CPU();

	void prepareBorder(double* result, double* source, int zStart, int zStop, int yStart, int yStop, int xStart,
			int xStop, int yCount, int xCount, int cellSize);

	void initState(double* state, initfunc_fill_ptr_t* userInitFuncs, unsigned short int* initFuncNumber,
			int blockNumber, double time);

	int getType();

	bool isCPU();
	bool isGPU();

	void copyArray(double* source, double* destination, int size);
	void copyArray(unsigned short int* source, unsigned short int* destination, int size);

	void sumArrays(double* result, double* arg1, double* arg2, int size);
	void multiplyArrayByNumber(double* result, double* arg, double factor, int size);
	void multiplyArrayByNumberAndSum(double* result, double* arg1, double factor, double* arg2, int size);

	double sumArrayElements(double* arg, int size);
	void maxElementsElementwise(double* result, double* arg1, double* arg2, int size);
	void divisionArraysElementwise(double* result, double* arg1, double* arg2, int size);

	void addNumberToArray(double* result, double* arg, double number, int size);
	void multiplyArraysElementwise(double* result, double* arg1, double* arg2, int size);

	void saveArray(double* array, int size, char* path);
	void loadArray(double* array, int size, std::ifstream& in);

	bool isNan(double* array, int size);

protected:
	double* getDoubleArray(int size);
	double** getDoublePointerArray(int size);

	int* getIntArray(int size);
	int** getIntPointerArray(int size);

	unsigned short int* getUnsignedShortIntArray(int size);

	void deallocDeviceSpecificArray(double* toDelete);
	void deallocDeviceSpecificArray(double** toDelete);

	void deallocDeviceSpecificArray(int* toDelete);
	void deallocDeviceSpecificArray(int** toDelete);

	void deallocDeviceSpecificArray(unsigned short int* toDelete);

	void printCell(double* array, int cellSize);
	void printArray1d(double* array, int xCount, int cellSize);
	void printArray2d(double* array, int yCount, int xCount, int cellSize);
	void printArray3d(double* array, int zCount, int yCount, int xCount, int cellSize);
};

#endif /* SRC_PROCESSINGUNIT_CPU_CPU_H_ */
