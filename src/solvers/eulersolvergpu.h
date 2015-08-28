/*
 * eulersolvergpu.h
 *
 *  Created on: 08 мая 2015 г.
 *      Author: frolov
 */

#ifndef SRC_SOLVERS_EULERSOLVERGPU_H_
#define SRC_SOLVERS_EULERSOLVERGPU_H_

#include "eulersolver.h"

class EulerSolverGpu: public EulerSolver {
public:
	EulerSolverGpu(int _count, double _aTol, double _rTol);
	virtual ~EulerSolverGpu();

	void copyState(double* result);
	void loadState(double* data);

	void prepareArgument(int stage, double timeStep);

	void print(int zCount, int yCount, int xCount, int cellSize) { std::cout << std::endl << "Solver Euler GPU print don't work" << std::endl; }
};

#endif /* SRC_SOLVERS_EULERSOLVERGPU_H_ */