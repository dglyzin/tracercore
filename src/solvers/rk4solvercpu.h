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
	RK4SolverCpu(int _count, double _aTol, double _rTol);
	virtual ~RK4SolverCpu();

	void copyState(double* result);

	void prepareArgument(int stage, double timeStep);

	void printToConsole(int zCount, int yCount, int xCount, int cellSize);
};

#endif /* SRC_SOLVERS_RK4SOLVERCPU_H_ */
