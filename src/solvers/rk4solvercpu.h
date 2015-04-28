/*
 * rk4solvercpu.h
 *
 *  Created on: 28 апр. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_SOLVERS_RK4SOLVERCPU_H_
#define SRC_SOLVERS_RK4SOLVERCPU_H_

#include "rk4solver.h"

class RK4SolverCpu: public RK4Solver {
public:
	RK4SolverCpu(int _count);
	virtual ~RK4SolverCpu();

	void copyState(double* result);

	void prepareArgument(int stage, double timeStep);
};

#endif /* SRC_SOLVERS_RK4SOLVERCPU_H_ */
