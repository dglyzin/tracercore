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
	DP45SolverGpu(int _count, double _aTol, double _rTol);
	virtual ~DP45SolverGpu();

	void copyState(double* result);
	void loadState(double* data);

	void prepareArgument(int stage, double timeStep);

	double getStepError(double timeStep);

	void print(int zCount, int yCount, int xCount, int cellSize) { std::cout << std::endl << "Solver DP45 GPU print don't work" << std::endl; }

	double* getDenseOutput(Solver* secondState) { std::cout << std::endl << "Solver DP45 GPU get dense output don't work" << std::endl; return NULL; }

private:
	void prepareFSAL(double timeStep);
};

#endif /* SRC_SOLVERS_DP45SOLVERGPU_H_ */
