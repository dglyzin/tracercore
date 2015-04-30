/*
 * dp45solvercpu.h
 *
 *  Created on: 30 апр. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_SOLVERS_DP45SOLVERCPU_H_
#define SRC_SOLVERS_DP45SOLVERCPU_H_

#include "dp45solver.h"

class DP45SolverCpu: public DP45Solver {
public:
	DP45SolverCpu(int _count);
	virtual ~DP45SolverCpu();

	void copyState(double* result);

	void prepareArgument(int stage, double timeStep);

	double getStepError(double timeStep, double aTol, double rTol);

private:
	void prepareFSAL(double timeStep);
};

#endif /* SRC_SOLVERS_DP45SOLVERCPU_H_ */
