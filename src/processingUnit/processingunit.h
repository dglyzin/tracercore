/*
 * processingunit.h
 *
 *  Created on: 18 сент. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_PROCESSINGUNIT_PROCESSINGUNIT_H_
#define SRC_PROCESSINGUNIT_PROCESSINGUNIT_H_

#include <list>

class ProcessingUnit {
public:
	ProcessingUnit();
	virtual ~ProcessingUnit();

	virtual void computeBorder(double* result, double** source, double time, double* parametrs, double** externalBorder) = 0;
	virtual void computeCenter(double* result, double** source, double time, double* parametrs, double** externalBorder) = 0;

	virtual void prepareBorder(double* result, double* source, int zStart, int zStop, int yStart, int yStop, int xStart, int xStop) = 0;

	virtual double* newDoubleArray(int size) = 0;
	virtual double** newDoublePointerArray(int size) = 0;

	virtual int* newIntArray(int size) = 0;
	virtual int** newIntPointerArray(int size) = 0;
};

#endif /* SRC_PROCESSINGUNIT_PROCESSINGUNIT_H_ */
