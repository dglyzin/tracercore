/*
 * dp45solver.h
 *
 *  Created on: 29 апр. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_SOLVERS_DP45SOLVER_H_
#define SRC_SOLVERS_DP45SOLVER_H_

#include "solver.h"

#include <algorithm>

class DP45Solver: public Solver {
public:
	DP45Solver();
	DP45Solver(int _count);
	virtual ~DP45Solver();

	virtual void copyState(double* result) { return; }

	double* getStageSource(int stage);
	double* getStageResult(int stage);

	virtual double getStageTimeStep(int stage);

	virtual void prepareArgument(int stage, double timeStep) { return; }

	void confirmStep(double timestep);
	void rejectStep(double timestep);

	virtual double getStepError(double timeStep, double aTol, double rTol) { return 0.0; }

    bool isFSAL() { return true; }
    bool isVariableStep() { return true; }
    int getStageCount() { return 6; }

	double getNewStep(double timeStep, double error, int totalDomainElements);
	bool isErrorPermissible(double error, int totalDomainElements);

	virtual void printToConsole(int zCount, int yCount, int xCount, int cellSize) { return; }

protected:
    double* mTempStore1;
    double* mTempStore2;
    double* mTempStore3;
    double* mTempStore4;
    double* mTempStore5;
    double* mTempStore6;
    double* mTempStore7;
    double* mArg;


    const double c2=0.2, c3=0.3, c4=0.8, c5=8.0/9.0;

    const double a21=0.2, a31=3.0/40.0, a32=9.0/40.0;
    const double a41=44.0/45.0, a42=-56.0/15.0, a43=32.0/9.0;
    const double a51=19372.0/6561.0, a52=-25360.0/2187.0;
    const double a53=64448.0/6561.0, a54=-212.0/729.0;
    const double a61=9017.0/3168.0, a62=-355.0/33.0, a63=46732.0/5247.0;
    const double a64=49.0/176.0, a65=-5103.0/18656.0;
    const double a71=35.0/384.0, a73=500.0/1113.0, a74=125.0/192.0;
    const double a75=-2187.0/6784.0, a76=11.0/84.0;
    const double e1=71.0/57600.0, e3=-71.0/16695.0, e4=71.0/1920.0;
    const double e5=-17253.0/339200.0, e6=22.0/525.0, e7=-1.0/40.0;
    const double facmin=0.5, facmax = 2, fac = 0.9;

    virtual void prepareFSAL(double timeStep) { return; }
};

#endif /* SRC_SOLVERS_DP45SOLVER_H_ */
