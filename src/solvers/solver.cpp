/*
 * solver.cpp
 *
 *  Created on: Feb 12, 2015
 *      Author: dglyzin
 */

#include <stdlib.h>
#include "solver.h"
#include <cassert>


/*Solver* GetCpuSolver(int solverIdx, int count){
	if      (solverIdx == EULER)
		return new EulerSolver(count);
	else if (solverIdx == RK4)
		return new EulerSolver(count);
	else
		return new EulerSolver(count);
}

Solver* GetGpuSolver(int solverIdx, int count){
	return NULL;
}*/



Solver::Solver(){

}

void Solver::copyState(double* result){
	for (int idx=0;idx<mCount;++idx)
		result[idx] = mState[idx];
}


EulerSolver::EulerSolver(int _count){
    mCount = _count;
    mState = new double[mCount];
    mTempStore1 = new double[mCount];
    for (int i = 0; i < mCount; ++i)
        mState[i]= 0;


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

double EulerSolver::getStageFactor(int stage, double timeStep){
    assert(stage == 0);
    return timeStep;
}

void EulerSolver::confirmStep(){
    double* temp = mState;
    mState = mTempStore1;
    mTempStore1 = temp;
}


