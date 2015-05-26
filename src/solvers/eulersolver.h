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
	EulerSolver();
	EulerSolver(int _count);
	~EulerSolver();

	virtual void copyState(double* result) { return; }

	double* getStageSource(int stage);
	double* getStageResult(int stage);

	double getStageTimeStep(int stage) { return 0.0; }

	void prepareArgument(int stage, double timeStep) { return; }

	void confirmStep(double timestep);
	void rejectStep(double timestep){};

	double getStepError(double timeStep, double aTol, double rTol) { return 0.0; }

    bool isFSAL() { return false; }
    bool isVariableStep() { return false; }
    int getStageCount() { return 1; }

	double getNewStep(double timeStep, double error, int totalDomainElements) { return timeStep; }
	bool isErrorPermissible(double error, int totalDomainElements) { return true; }

	virtual void printToConsole(int zCount, int yCount, int xCount, int cellSize) { return; }

protected:
    double* mTempStore1;

};

#endif /* SRC_SOLVERS_EULERSOLVER_H_ */
