/*
 * rk4solver.h
 *
 *  Created on: 28 апр. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_SOLVERS_RK4SOLVER_H_
#define SRC_SOLVERS_RK4SOLVER_H_

#include "solver.h"

class RK4Solver: public Solver {
public:
public:
	RK4Solver(int _mCount);
	~RK4Solver();

    virtual void copyState(double* result) { return; }

	double* getStageSource(int stage);
	double* getStageResult(int stage);

	double getStageTimeStep(int stage);

	void prepareArgument(int stage, double timeStep) { return; }

	void confirmStep(double timestep);

	double getStepError(double timeStep, double aTol, double rTol) { return 0.0; }

protected:
    double* mTempStore1;
    double* mTempStore2;
    double* mTempStore3;
    double* mTempStore4;
    double* mArg;

    const double b1 = 1.0/6.0;
    const double b2 = 1.0/3.0;
    const double b3 = 1.0/3.0;
    const double b4 = 1.0/6.0;
};

#endif /* SRC_SOLVERS_RK4SOLVER_H_ */
