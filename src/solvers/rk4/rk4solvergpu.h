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
	RK4SolverGpu(int _count, double _aTol, double _rTol);
	virtual ~RK4SolverGpu();

	void copyState(double* result);
	void loadState(double* data);

	void prepareArgument(int stage, double timeStep);

	void print(int zCount, int yCount, int xCount, int cellSize) { std::cout << std::endl << "Solver RK4 GPU print don't work" << std::endl; }
};

#endif /* RK4SOLVERGPU_H_ */