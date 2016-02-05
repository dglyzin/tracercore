/*
 * cpu.h
 *
 *  Created on: 18 сент. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_PROCESSINGUNIT_CPU_CPU_H_
#define SRC_PROCESSINGUNIT_CPU_CPU_H_

#include <omp.h>

#include "../processingunit.h"

class CPU: public ProcessingUnit {
public:
	CPU(int _deviceNumber);
	virtual ~CPU();

	//virtual void computeBorder(double* result, double** source, double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize) = 0;
	//virtual void computeCenter(double* result, double** source, double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize) = 0;

	void prepareBorder(double* result, double* source, int zStart, int zStop, int yStart, int yStop, int xStart, int xStop, int yCount, int xCount, int cellSize);

	void initState(double* state, initfunc_fill_ptr_t* userInitFuncs, unsigned short int* initFuncNumber, int blockNumber, double time);

	int getType();

	bool isCPU();
	bool isGPU();


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


	void saveArray(double* array, std::ofstream& out);
	void loadArray(double* array, std::ifstream& in);
};

#endif /* SRC_PROCESSINGUNIT_CPU_CPU_H_ */
