/*
 * rk4solvergpu.h
 *
 *  Created on: 13 мая 2015 г.
 *      Author: frolov
 */

#ifndef RK4SOLVERGPU_H_
#define RK4SOLVERGPU_H_

#include "rk4solver.h"

class RK4SolverGpu: public RK4Solver {
public:
	RK4SolverGpu(int _count);
	virtual ~RK4SolverGpu();

	void copyState(double* result);

	void prepareArgument(int stage, double timeStep);
};

#endif /* RK4SOLVERGPU_H_ */
