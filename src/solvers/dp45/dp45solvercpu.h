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
	DP45SolverCpu(int _count, double _aTol, double _rTol);
	virtual ~DP45SolverCpu();

	void copyState(double* result);
	void loadState(double* data);

	void prepareArgument(int stage, double timeStep);

	double getStepError(double timeStep);

	void print(int zCount, int yCount, int xCount, int cellSize);

	double* getDenseOutput(Solver* secondState);

private:
	void prepareFSAL(double timeStep);
};

#endif /* SRC_SOLVERS_DP45SOLVERCPU_H_ */
