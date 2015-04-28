/*
 * rk4solver.cpp
 *
 *  Created on: 28 апр. 2015 г.
 *      Author: frolov
 */

#include "rk4solver.h"

RK4Solver::RK4Solver(int _count) : Solver(_count){
    mCount = _count;
    mTempStore1 = mTempStore2 = mTempStore3 = mTempStore4 = mArg = NULL;
    /*mState = new double[mCount];
    mTempStore1 = new double[mCount];
    mTempStore2 = new double[mCount];
    mTempStore3 = new double[mCount];
    mTempStore4 = new double[mCount];
    mArg = new double[mCount];
    for (int idx = 0; idx < mCount; ++idx){
        mState[idx] = 0;
        mTempStore1[idx] = 0;
    }*/
}

RK4Solver::~RK4Solver(){
    /*delete mState;
    delete mTempStore1;
    delete mTempStore2;
    delete mTempStore3;
    delete mTempStore4;
    delete mArg;*/
}

void RK4Solver::prepareArgument(int stage, double timeStep) {

	if      (stage == 0)
#pragma omp parallel for
		for (int idx = 0; idx < mCount; idx++)
			mArg[idx] = mState[idx] + 0.5 * timeStep*mTempStore1[idx];
	else if (stage == 1)
#pragma omp parallel for
		for (int idx = 0; idx < mCount; idx++)
			mArg[idx] = mState[idx] + 0.5 * timeStep*mTempStore2[idx];
	else if (stage == 2)
#pragma omp parallel for
		for (int idx = 0; idx < mCount; idx++)
			mArg[idx] = mState[idx] + timeStep * mTempStore3[idx];
	else if (stage == 3){
	    const double b1 = 1.0/6.0;
	    const double b2 = 1.0/3.0;
	    const double b3 = 1.0/3.0;
	    const double b4 = 1.0/6.0;
	#pragma omp parallel for
		for (int idx = 0; idx < mCount; idx++)
			mArg[idx] = mState[idx] + timeStep * ( b1 * mTempStore1[idx] + b2 * mTempStore2[idx] + b3 * mTempStore3[idx] + b4 * mTempStore4[idx] );
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

// TODO на что влияет?
double RK4Solver::getStageTimeStep(int stage){
    if      (stage == 0) return 0.0;
    else if (stage == 1) return 0.5;
    else if (stage == 2) return 0.5;
    else if (stage == 3) return 1.0;
    else assert(0);
    return 0.0;
}

void RK4Solver::confirmStep(double timestep){
    double* temp = mState;
    mState = mArg;
    mArg = temp;
}
