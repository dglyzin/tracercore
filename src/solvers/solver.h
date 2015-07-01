/*
 * solver.h
 *
 *  Created on: Feb 12, 2015
 *      Author: dglyzin
 */

#ifndef SOLVER_H_
#define SOLVER_H_

#include <cassert>

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <math.h>
#include <string.h>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "../cuda_func.h"

class Solver {
public:
	Solver();
    Solver(int _count, double _aTol, double _rTol);
    virtual ~Solver() { return; }

    virtual void copyState(double* result) = 0;
    virtual void loadState(double* data) = 0;

    double* getStatePtr(){ return mState;}

    virtual double* getStageSource(int stage) = 0;
    virtual double* getStageResult(int stage) = 0;

    virtual double getStageTimeStep(int stage) = 0;

    virtual void prepareArgument(int stage, double timeStep) = 0;

    virtual void confirmStep(double timestep) = 0;
    virtual void rejectStep(double timestep) = 0;

    virtual double getStepError(double timeStep) = 0;

    virtual bool isFSAL() = 0;
    virtual bool isVariableStep() = 0;
    virtual int getStageCount() = 0;

	virtual double getNewStep(double timeStep, double error, int totalDomainElements) = 0;
	virtual bool isErrorPermissible(double error, int totalDomainElements) = 0;

	virtual void print(int zCount, int yCount, int xCount, int cellSize) = 0;

protected:
  	int     mCount; //total number of elements in every array
  	double* mState;

  	double aTol;
  	double rTol;

  	void printMatrix(double* matrix, int zCount, int yCount, int xCount, int cellSize);
};

#endif /* SOLVER_H_ */
