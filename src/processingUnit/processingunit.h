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

#include <list>

class ProcessingUnit {
public:
	ProcessingUnit();
	virtual ~ProcessingUnit();

	virtual void computeBorder(double* result, double** source, double time, double* parametrs, double** externalBorder) = 0;
	virtual void computeCenter(double* result, double** source, double time, double* parametrs, double** externalBorder) = 0;

	virtual void prepareBorder(double* result, double* source, int zStart, int zStop, int yStart, int yStop, int xStart, int xStop) = 0;

	double* newDoubleArray(int size);
	double** newDoublePointerArray(int size);

	int* newIntArray(int size);
	int** newIntPointerArray(int size);

	double* newDoublePinnedArray(int size);

private:
	std::list<double*> doubleArrays;
	std::list<double**> doublePointerArrays;

	std::list<int*> intArrays;
	std::list<int**> intPointerArrays;

	std::list<double*> doublePinnedArrays;



	void deleteAllArrays();

	void deleteAllDoubleArrays();
	void deleteAllDoublePointerArrays();

	void deleteAllIntArrays();
	void deleteAllIntPonterArrays();

	void deleteAllDoublePinnedArrays();



	virtual double* getDoubleArray(int size) = 0;
	virtual double** getDoublePointerArray(int size) = 0;

	virtual int* getIntArray(int size) = 0;
	virtual int** getIntPointerArray(int size) = 0;

	double* getDoublePinnedArray(int size);



	virtual void deleteIntArray(int* toDelete) = 0;
	virtual void deleteIntPointerArray(int** toDelete) = 0;

	virtual void deleteDoubleArray(double* toDelete) = 0;
	virtual void deleteDoublePointerArray(double** toDelete) = 0;

	void deleteDoublePinnedArray(double* toDelete);
};

#endif /* SRC_PROCESSINGUNIT_PROCESSINGUNIT_H_ */
