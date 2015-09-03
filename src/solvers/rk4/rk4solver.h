/*
 * rk4solver.h
 *
 *  Created on: 28 апр. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_SOLVERS_RK4SOLVER_H_
#define SRC_SOLVERS_RK4SOLVER_H_

#include "../solver.h"

class RK4Solver: public Solver {
public:
public:
	RK4Solver();
	RK4Solver(int _count, double _aTol, double _rTol);
	~RK4Solver();

    virtual void copyState(double* result) { return; }
    virtual void loadState(double* data) { return; }

	double* getStageSource(int stage);
	double* getStageResult(int stage);

	double getStageTimeStep(int stage);

	virtual void prepareArgument(int stage, double timeStep) { return; }

	void confirmStep(double timestep);
	void rejectStep(double timestep){};

	double getStepError(double timeStep) { return 0.0; }

    bool isFSAL() { return false; }
    bool isVariableStep() { return false; }
    int getStageCount() { return 4; }

	double getNewStep(double timeStep, double error, int totalDomainElements) { return timeStep; }
	bool isErrorPermissible(double error, int totalDomainElements) { return true; }

	virtual void print(int zCount, int yCount, int xCount, int cellSize) { return; }

	virtual double* getDencseOutput(Solver* secondState) { return NULL; }

protected:
    double* mTempStore1;
    double* mTempStore2;
    double* mTempStore3;
    double* mTempStore4;
    double* mArg;

    static const double b1 = 1.0/6.0;
    static const double b2 = 1.0/3.0;
    static const double b3 = 1.0/3.0;
    static const double b4 = 1.0/6.0;
};

#endif /* SRC_SOLVERS_RK4SOLVER_H_ */
