/*
 * eulersolvercpu.h
 *
 *  Created on: 28 апр. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_SOLVERS_EULERSOLVERCPU_H_
#define SRC_SOLVERS_EULERSOLVERCPU_H_

#include "eulersolver.h"

class EulerSolverCpu: public EulerSolver {
public:
	EulerSolverCpu(int _count, double _aTol, double _rTol);
	virtual ~EulerSolverCpu();

	void copyState(double* result);
	void loadState(double* data);

	void prepareArgument(int stage, double timeStep);

	double* getMState();

	void print(int zCount, int yCount, int xCount, int cellSize);
};

#endif /* SRC_SOLVERS_EULERSOLVERCPU_H_ */