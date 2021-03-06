/*
 * cpu2d.h
 *
 *  Created on: 23 сент. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_PROCESSINGUNIT_CPU_CPU2D_H_
#define SRC_PROCESSINGUNIT_CPU_CPU2D_H_

#include "../../processingunit/cpu/cpu.h"

class CPU_2d: public CPU {
public:
	CPU_2d(int _deviceNumber);
	virtual ~CPU_2d();

	void computeBorder(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result, double** source,
			double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize);
	void computeCenter(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result, double** source,
			double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize);

	void printArray(double* array, int zCount, int yCount, int xCount, int cellSize);
};

#endif /* SRC_PROCESSINGUNIT_CPU_CPU2D_H_ */
