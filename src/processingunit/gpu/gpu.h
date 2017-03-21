/*
 * gpu.h
 *
 *  Created on: 25 марта 2016 г.
 *      Author: frolov
 */

#ifndef SRC_PROCESSINGUNIT_GPU_GPU_H_
#define SRC_PROCESSINGUNIT_GPU_GPU_H_

#include "../processingunit.h"
#include "../../cuda_func.h"

class GPU: public ProcessingUnit {
public:
	GPU(int _deviceNumber);
	virtual ~GPU();

	void prepareBorder(double* result, double* source, int zStart, int zStop, int yStart, int yStop, int xStart,
			int xStop, int yCount, int xCount, int cellSize);

	void initState(double* state, initfunc_fill_ptr_t* userInitFuncs, unsigned short int* initFuncNumber,
			int blockNumber, double time);

	int getType();

	bool isCPU();
	bool isGPU();

	bool isDeviceType(int type);

	void swapStorages(double** firstArray, double** secondArray);

	void copyArray(double* source, double* destination, int size);
	void copyArray(unsigned short int* source, unsigned short int* destination, int size);

	void sumArrays(double* result, double* arg1, double* arg2, int size);
	void multiplyArrayByNumber(double* result, double* arg, double factor, int size);
	void multiplyArrayByNumberAndSum(double* result, double* arg1, double factor, double* arg2, int size);

	double sumArrayElements(double* arg, int size);
	void maxElementsElementwise(double* result, double* arg1, double* arg2, int size);
	void maxAbsElementsElementwise(double* result, double* arg1, double* arg2, int size);
	void divisionArraysElementwise(double* result, double* arg1, double* arg2, int size);

	void addNumberToArray(double* result, double* arg, double number, int size);
	void multiplyArraysElementwise(double* result, double* arg1, double* arg2, int size);

	/*void saveArray(double* array, int size, char* path);
	void loadArray(double* array, int size, std::ifstream& in);*/

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

	void writeArray(double* array, int size, std::ofstream& out);
	void readArray(double* array, int size, std::ifstream& in);

	/*void printCell(double* array, int cellSize);
	void printArray1d(double* array, int xCount, int cellSize);
	void printArray2d(double* array, int yCount, int xCount, int cellSize);
	void printArray3d(double* array, int zCount, int yCount, int xCount, int cellSize);*/
};

#endif /* SRC_PROCESSINGUNIT_GPU_GPU_H_ */
