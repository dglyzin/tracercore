/*
 * solver.h
 *
 *  Created on: Feb 12, 2015
 *      Author: dglyzin
 */

#ifndef SOLVER_H_
#define SOLVER_H_

#include <stdlib.h>
#include <omp.h>

#include <stdlib.h>
#include <cassert>
#include <math.h>

#include <cuda.h>
#include <cuda_runtime_api.h>

class Solver {
public:
	Solver();
    Solver(int _count);
    virtual ~Solver() { return; }

    virtual void copyState(double* result) = 0;

    double* getStatePtr(){ return mState;}

    virtual double* getStageSource(int stage) = 0;
    virtual double* getStageResult(int stage) = 0;

    virtual double getStageTimeStep(int stage) = 0;

    virtual void prepareArgument(int stage, double timeStep) = 0;

    virtual void confirmStep(double timestep) = 0;

    virtual double getStepError(double timeStep, double aTol, double rTol) = 0;

    virtual bool isFSAL() = 0;
    virtual bool isVariableStep() = 0;
    virtual int getStageCount() = 0;

	virtual double getNewStep(double timeStep, double error, int totalDomainElements) = 0;
	virtual bool isErrorPermissible(double error, int totalDomainElements) = 0;

protected:
  	int     mCount; //total number of elements in every array
  	double* mState;
};

#endif /* SOLVER_H_ */
