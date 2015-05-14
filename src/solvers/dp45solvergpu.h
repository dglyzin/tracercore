/*
 * dp45solvergpu.h
 *
 *  Created on: 14 мая 2015 г.
 *      Author: frolov
 */

#ifndef SRC_SOLVERS_DP45SOLVERGPU_H_
#define SRC_SOLVERS_DP45SOLVERGPU_H_

#include "dp45solver.h"

class DP45SolverGpu: public DP45Solver {
public:
	DP45SolverGpu(int _count);
	virtual ~DP45SolverGpu();

	void copyState(double* result);

	void prepareArgument(int stage, double timeStep);

	double getStepError(double timeStep, double aTol, double rTol);

private:
	void prepareFSAL(double timeStep);
};

#endif /* SRC_SOLVERS_DP45SOLVERGPU_H_ */
