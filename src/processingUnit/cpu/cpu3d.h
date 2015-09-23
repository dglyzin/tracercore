/*
 * cpu3d.h
 *
 *  Created on: 23 сент. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_PROCESSINGUNIT_CPU_CPU3D_H_
#define SRC_PROCESSINGUNIT_CPU_CPU3D_H_

#include "cpu.h"

class CPU_3d: public CPU {
public:
	CPU_3d();
	virtual ~CPU_3d();

	void computeBorder(double* result, double** source, double time, double* parametrs, double** externalBorder);
	void computeCenter(double* result, double** source, double time, double* parametrs, double** externalBorder);
};

#endif /* SRC_PROCESSINGUNIT_CPU_CPU3D_H_ */
