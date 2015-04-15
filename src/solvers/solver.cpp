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

SolverInfo::SolverInfo(){

}

void Solver::copyState(double* result){
	for (int idx=0;idx<mCount;++idx)
		result[idx] = mState[idx];
}

//****************1. EULER SOLVER*************//
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

void EulerSolver::confirmStep(){
    double* temp = mState;
    mState = mTempStore1;
    mTempStore1 = temp;
}

//****************2. RK4 SOLVER*************//
/*
 * Stage 0: mState -f-> ts1, mState+h/2 ts1 --> arg
 * Stage 1: arg -f-> ts2, mState+h/2 ts2 --> arg
 * Stage 2: arg -f-> ts3, mState+h ts3   --> arg
 * Stage 3: arg -f-> ts4, mState+h/6 ts1 + h/3 ts2 + h/3 ts3 + h/4 ts4 --> arg
 *
 * confirm: mState<-->arg
 */

RK4Solver::RK4Solver(int _count){
    mCount = _count;
    mState = new double[mCount];
    mTempStore1 = new double[mCount];
    mTempStore2 = new double[mCount];
    mTempStore3 = new double[mCount];
    mTempStore4 = new double[mCount];
    mArg = new double[mCount];
    for (int idx = 0; idx < mCount; ++idx){
        mState[idx] = 0;
        mTempStore1[idx] = 0;
    }
}

RK4Solver::~RK4Solver(){
    delete mState;
    delete mTempStore1;
    delete mTempStore2;
    delete mTempStore3;
    delete mTempStore4;
    delete mArg;
}

void RK4Solver::prepareArgument(int stage, double timeStep) {

	if      (stage == 0)
#pragma omp parallel for
		for (int idx=0; idx<mCount; idx++)
			mArg[idx] = mState[idx]+0.5*timeStep*mTempStore1[idx];
	else if (stage == 1)
#pragma omp parallel for
		for (int idx=0; idx<mCount; idx++)
			mArg[idx] = mState[idx]+0.5*timeStep*mTempStore2[idx];
	else if (stage == 2)
#pragma omp parallel for
		for (int idx=0; idx<mCount; idx++)
			mArg[idx] = mState[idx]+timeStep*mTempStore3[idx];
	else if (stage == 3){
	    const double b1 = 1.0/6.0;
	    const double b2 = 1.0/3.0;
	    const double b3 = 1.0/3.0;
	    const double b4= 1.0/6.0;
#pragma omp parallel for
		for (int idx=0; idx<mCount; idx++)
			mArg[idx] = mState[idx]+timeStep*(b1*mTempStore1[idx]+b2*mTempStore2[idx]+b3*mTempStore3[idx]+b4*mTempStore4[idx]);
	}
	else assert(0);
}

double* RK4Solver::getStageSource(int stage){
	if      (stage == 0) return mState;
	else if (stage == 1) return mArg;
	else if (stage == 2) return mArg;
	else if (stage == 3) return mArg;
	else assert(0);
	return NULL;
}

double* RK4Solver::getStageResult(int stage){
	if      (stage == 0) return mTempStore1;
	else if (stage == 1) return mTempStore2;
	else if (stage == 2) return mTempStore3;
	else if (stage == 3) return mTempStore4;
	else assert(0);
	return NULL;
}


double RK4Solver::getStageTimeStep(int stage){
    if      (stage == 0) return 0.0;
    else if (stage == 1) return 0.5;
    else if (stage == 2) return 0.5;
    else if (stage == 3) return 1.0;
    else assert(0);
    return 0.0;
}

void RK4Solver::confirmStep(){
    double* temp = mState;
    mState = mArg;
    mArg = temp;
}




//****************3. DP45 SOLVER*************//
/*
 * Stage 0: mState -f-> ts1, mState+h/2 ts1 --> arg
 * Stage 1: arg -f-> ts2, mState+h/2 ts2 --> arg
 * Stage 2: arg -f-> ts3, mState+h ts3   --> arg
 * Stage 3: arg -f-> ts4, mState+h/6 ts1 + h/3 ts2 + h/3 ts3 + h/4 ts4 --> arg
 *
 * confirm: mState<-->arg
 */

DP45Solver::DP45Solver(int _count){
    mCount = _count;
    mState = new double[mCount];
    mTempStore1 = new double[mCount];
    mTempStore2 = new double[mCount];
    mTempStore3 = new double[mCount];
    mTempStore4 = new double[mCount];
    mArg = new double[mCount];
    for (int idx = 0; idx < mCount; ++idx){
        mState[idx] = 0;
        mTempStore1[idx] = 0;
    }
}

DP45Solver::~DP45Solver(){
    delete mState;
    delete mTempStore1;
    delete mTempStore2;
    delete mTempStore3;
    delete mTempStore4;
    delete mArg;
}

void DP45Solver::prepareArgument(int stage, double timeStep) {

	if      (stage == 0)
#pragma omp parallel for
		for (int idx=0; idx<mCount; idx++)
			mArg[idx] = mState[idx]+0.5*timeStep*mTempStore1[idx];
	else if (stage == 1)
#pragma omp parallel for
		for (int idx=0; idx<mCount; idx++)
			mArg[idx] = mState[idx]+0.5*timeStep*mTempStore2[idx];
	else if (stage == 2)
#pragma omp parallel for
		for (int idx=0; idx<mCount; idx++)
			mArg[idx] = mState[idx]+timeStep*mTempStore3[idx];
	else if (stage == 3){
	    const double b1 = 1.0/6.0;
	    const double b2 = 1.0/3.0;
	    const double b3 = 1.0/3.0;
	    const double b4= 1.0/6.0;
#pragma omp parallel for
		for (int idx=0; idx<mCount; idx++)
			mArg[idx] = mState[idx]+timeStep*(b1*mTempStore1[idx]+b2*mTempStore2[idx]+b3*mTempStore3[idx]+b4*mTempStore4[idx]);
	}
	else assert(0);
}

double* DP45Solver::getStageSource(int stage){
	if      (stage == 0) return mState;
	else if (stage == 1) return mArg;
	else if (stage == 2) return mArg;
	else if (stage == 3) return mArg;
	else assert(0);
	return NULL;
}

double* DP45Solver::getStageResult(int stage){
	if      (stage == 0) return mTempStore1;
	else if (stage == 1) return mTempStore2;
	else if (stage == 2) return mTempStore3;
	else if (stage == 3) return mTempStore4;
	else assert(0);
	return NULL;
}


double DP45Solver::getStageTimeStep(int stage){
    if      (stage == 0) return 0.0;
    else if (stage == 1) return 0.5;
    else if (stage == 2) return 0.5;
    else if (stage == 3) return 1.0;
    else assert(0);
    return 0.0;
}

void DP45Solver::confirmStep(){
    double* temp = mState;
    mState = mTempStore1;
    mTempStore1 = temp;
}
double DP45Solver::getStepError(){
	return 0.0;
}
