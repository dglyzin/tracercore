/*
 * eulersolvercpu.h
 *
 *  Created on: 28 апр. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_SOLVERS_EULERSOLVERCPU_H_
#define SRC_SOLVERS_EULERSOLVERCPU_H_

#include "solver.h"

class EulerSolverCpu: public EulerSolver {
public:
	EulerSolverCpu(int _mCount);
	virtual ~EulerSolverCpu();

	void copyState(double* result);

	void prepareArgument(int stage, double timeStep);
};

#endif /* SRC_SOLVERS_EULERSOLVERCPU_H_ */
