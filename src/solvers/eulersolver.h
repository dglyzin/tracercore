/*
 * eulersolver.h
 *
 *  Created on: 28 апр. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_SOLVERS_EULERSOLVER_H_
#define SRC_SOLVERS_EULERSOLVER_H_

#include "solver.h"

class EulerSolver: public Solver{
public:
	EulerSolver(int _count);
	~EulerSolver();

	virtual void copyState(double* result) { return; }

	double* getStageSource(int stage);
	double* getStageResult(int stage);

	virtual double getStageTimeStep(int stage) { return 0; }

	void prepareArgument(int stage, double timeStep) { return; }

	void confirmStep(double timestep);

	double getStepError(double timeStep, double aTol, double rTol) { return 0.0; }

protected:
    double* mTempStore1;

};

#endif /* SRC_SOLVERS_EULERSOLVER_H_ */
