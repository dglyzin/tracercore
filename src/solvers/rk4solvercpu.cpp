/*
 * rk4solvercpu.cpp
 *
 *  Created on: 28 апр. 2015 г.
 *      Author: frolov
 */

#include "rk4solvercpu.h"

RK4SolverCpu::RK4SolverCpu(int _count) : RK4Solver(_count) {
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

RK4SolverCpu::~RK4SolverCpu() {
    delete mState;
    delete mTempStore1;
    delete mTempStore2;
    delete mTempStore3;
    delete mTempStore4;
    delete mArg;
}

void RK4SolverCpu::copyState(double* result) {
	for (int idx = 0; idx < mCount; ++idx)
		result[idx] = mState[idx];
}

void RK4SolverCpu::prepareArgument(int stage, double timeStep) {

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

	#pragma omp parallel for
		for (int idx = 0; idx < mCount; idx++)
			mArg[idx] = mState[idx] + timeStep * ( b1 * mTempStore1[idx] + b2 * mTempStore2[idx] + b3 * mTempStore3[idx] + b4 * mTempStore4[idx] );
	}
	else assert(0);
}
