/*
 * cpu1d.h
 *
 *  Created on: 23 сент. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_PROCESSINGUNIT_CPU_CPU1D_H_
#define SRC_PROCESSINGUNIT_CPU_CPU1D_H_

#include "../../processingunit/cpu/cpu.h"

class CPU_1d: public CPU {
public:
	CPU_1d(int _nodeNumber, int _deviceNumber);
	virtual ~CPU_1d();

	void computeBorder(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result, double** source, double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize);
	void computeCenter(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result, double** source, double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize);
};
#endif /* SRC_PROCESSINGUNIT_CPU_CPU1D_H_ */

