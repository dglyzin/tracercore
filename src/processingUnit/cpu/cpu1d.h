/*
 * cpu1d.h
 *
 *  Created on: 23 сент. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_PROCESSINGUNIT_CPU_CPU1D_H_
#define SRC_PROCESSINGUNIT_CPU_CPU1D_H_

#include "cpu.h"

class CPU_1d: public CPU {
public:
	CPU_1d();
	virtual ~CPU_1d();

	void computeBorder(double* result, double** source, double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize, double* mParams);
	void computeCenter(double* result, double** source, double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize, double* mParams);
};
#endif /* SRC_PROCESSINGUNIT_CPU_CPU1D_H_ */

