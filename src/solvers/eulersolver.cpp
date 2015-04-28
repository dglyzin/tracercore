/*
 * eulersolver.cpp
 *
 *  Created on: 28 апр. 2015 г.
 *      Author: frolov
 */

#include "eulersolver.h"

EulerSolver::EulerSolver(int _count){
    mCount = _count;
    mState = new double[mCount];
    mTempStore1 = new double[mCount];
    for (int idx = 0; idx < mCount; ++idx){
        mState[idx] = 0;
        mTempStore1[idx] = 0;
    }
}

EulerSolver::~EulerSolver(){
    delete mState;
    delete mTempStore1;
}

double* EulerSolver::getStageSource(int stage){
    assert(stage == 0);
    return mState;
}

double* EulerSolver::getStageResult(int stage){
    assert(stage == 0);
    return mTempStore1;
}

void EulerSolver::prepareArgument(int stage, double timeStep){
#pragma omp parallel for
	for (int idx = 0; idx < mCount; ++idx)
	    mTempStore1[idx]= mState[idx] + timeStep*mTempStore1[idx];
}

void EulerSolver::confirmStep(double timestep){
    double* temp = mState;
    mState = mTempStore1;
    mTempStore1 = temp;
}
