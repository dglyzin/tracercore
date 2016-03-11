/*
 * cpu3d.h
 *
 *  Created on: 23 сент. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_PROCESSINGUNIT_CPU_CPU3D_H_
#define SRC_PROCESSINGUNIT_CPU_CPU3D_H_

#include "../../processingunit/cpu/cpu.h"

class CPU_3d: public CPU {
public:
	CPU_3d(int _deviceNumber);
	virtual ~CPU_3d();

	void computeBorder(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result, double** source,
			double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize);
	void computeCenter(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result, double** source,
			double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize);

	void printArray(double* array, int zCount, int yCount, int xCount, int haloSize);
};

#endif /* SRC_PROCESSINGUNIT_CPU_CPU3D_H_ */
