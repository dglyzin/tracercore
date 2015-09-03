/*
 * eulersolver.h
 *
 *  Created on: 28 апр. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_SOLVERS_EULERSOLVER_H_
#define SRC_SOLVERS_EULERSOLVER_H_

#include "../solver.h"

class EulerSolver: public Solver{
public:
	EulerSolver();
	EulerSolver(int _count, double _aTol, double _rTol);
	~EulerSolver();

	virtual void copyState(double* result) { return; }
	virtual void loadState(double* data) { return; }

	double* getStageSource(int stage);
	double* getStageResult(int stage);

	double getStageTimeStep(int stage) { return 0.0; }

	void prepareArgument(int stage, double timeStep) { return; }

	void confirmStep(double timestep);
	void rejectStep(double timestep){};

	double getStepError(double timeStep) { return 0.0; }

    bool isFSAL() { return false; }
    bool isVariableStep() { return false; }
    int getStageCount() { return 1; }

	double getNewStep(double timeStep, double error, int totalDomainElements) { return timeStep; }
	bool isErrorPermissible(double error, int totalDomainElements) { return true; }

	virtual void print(int zCount, int yCount, int xCount, int cellSize) { return; }

	virtual double* getDenseOutput(Solver* secondState) { return NULL; }

protected:
    double* mTempStore1;

};

#endif /* SRC_SOLVERS_EULERSOLVER_H_ */
