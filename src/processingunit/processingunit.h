/*
 * processingunit.h
 *
 *  Created on: 18 сент. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_PROCESSINGUNIT_PROCESSINGUNIT_H_
#define SRC_PROCESSINGUNIT_PROCESSINGUNIT_H_

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdlib.h>
#include <stdio.h>

#include <list>

#include <fstream>

#include "../enums.h"
#include "../logger.h"

#include "../userfuncs.h"

class ProcessingUnit {
public:
	ProcessingUnit(int _deviceNumber);
	virtual ~ProcessingUnit();

	virtual void computeBorder(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result,
			double** source, double time, double* parametrs, double** externalBorder, int zCount, int yCount,
			int xCount, int haloSize) = 0;
	virtual void computeCenter(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result,
			double** source, double time, double* parametrs, double** externalBorder, int zCount, int yCount,
			int xCount, int haloSize) = 0;

	virtual void prepareBorder(double* result, double* source, int zStart, int zStop, int yStart, int yStop, int xStart,
			int xStop, int yCount, int xCount, int cellSize) = 0;

	virtual void initState(double* state, initfunc_fill_ptr_t* userInitFuncs, unsigned short int* initFuncNumber,
			int blockNumber, double time) = 0;

	virtual int getType() = 0;

	virtual bool isCPU() = 0;
	virtual bool isGPU() = 0;

	virtual bool isDeviceType(int type) = 0;
	bool isDeviceNumber(int number);

	int getDeviceNumber();

	double* newDoubleArray(int size);
	double** newDoublePointerArray(int size);

	int* newIntArray(int size);
	int** newIntPointerArray(int size);

	double* newDoublePinnedArray(int size);

	unsigned short int* newUnsignedShortIntArray(int size);

	void deleteDeviceSpecificArray(double* toDelete);
	void deleteDeviceSpecificArray(double** toDelete);

	void deleteDeviceSpecificArray(int* toDelete);
	void deleteDeviceSpecificArray(int** toDelete);

	void deleteDoublePinnedArray(double* toDelete);

	void deleteUnsignedShortInt(unsigned short int* toDelete);

	virtual void swapStorages(double** firstArray, double** secondArray) = 0;

	virtual void copyArray(double* source, double* destination, int size) = 0;
	virtual void copyArray(unsigned short int* source, unsigned short int* destination, int size) = 0;

	virtual void sumArrays(double* result, double* arg1, double* arg2, int size) = 0;
	virtual void multiplyArrayByNumber(double* result, double* arg, double factor, int size) = 0;
	virtual void multiplyArrayByNumberAndSum(double* result, double* arg1, double factor, double* arg2, int size) = 0;

	virtual double sumArrayElements(double* arg, int size) = 0;
	virtual void maxElementsElementwise(double* result, double* arg1, double* arg2, int size) = 0;
	virtual void maxAbsElementsElementwise(double* result, double* arg1, double* arg2, int size) = 0;
	virtual void divisionArraysElementwise(double* result, double* arg1, double* arg2, int size) = 0;

	virtual void addNumberToArray(double* result, double* arg, double number, int size) = 0;
	virtual void multiplyArraysElementwise(double* result, double* arg1, double* arg2, int size) = 0;

	void saveArray(double* array, int size, char* path);
	void loadArray(double* array, int size, std::ifstream& in);

	virtual bool isNan(double* array, int size) = 0;

	virtual void printArray(double* array, int zCount, int yCount, int xCount, int cellSize) = 0;

protected:
	int deviceNumber;

	std::list<double*> doubleArrays;
	std::list<double**> doublePointerArrays;

	std::list<int*> intArrays;
	std::list<int**> intPointerArrays;

	std::list<double*> doublePinnedArrays;

	std::list<unsigned short int*> unsignedShortIntArrays;

	virtual double* getDoubleArray(int size) = 0;
	virtual double** getDoublePointerArray(int size) = 0;

	virtual int* getIntArray(int size) = 0;
	virtual int** getIntPointerArray(int size) = 0;

	double* getDoublePinnedArray(int size);

	virtual unsigned short int* getUnsignedShortIntArray(int size) = 0;

	virtual void deallocDeviceSpecificArray(double* toDelete) = 0;
	virtual void deallocDeviceSpecificArray(double** toDelete) = 0;

	virtual void deallocDeviceSpecificArray(int* toDelete) = 0;
	virtual void deallocDeviceSpecificArray(int** toDelete) = 0;

	void deallocDoublePinnedArray(double* toDelete);

	virtual void deallocDeviceSpecificArray(unsigned short int* toDelete) = 0;

	void deleteAllArrays();

	void deleteAllDoubleArrays();
	void deleteAllDoublePointerArrays();

	void deleteAllIntArrays();
	void deleteAllIntPonterArrays();

	void deleteAllDoublePinnedArrays();

	void deleteAllUnsignedShortInt();

	virtual void writeArray(double* array, int size, std::ofstream& out) = 0;
	virtual void readArray(double* array, int size, std::ifstream& in) = 0;

	void printCell(double* array, int cellSize);
	void printArray1d(double* array, int xCount, int cellSize);
	void printArray2d(double* array, int yCount, int xCount, int cellSize);
	void printArray3d(double* array, int zCount, int yCount, int xCount, int cellSize);
};

#endif /* SRC_PROCESSINGUNIT_PROCESSINGUNIT_H_ */
