/*
 * rk4solvergpu.cpp
 *
 *  Created on: 13 мая 2015 г.
 *      Author: frolov
 */

#include "rk4solvergpu.h"

RK4SolverGpu::RK4SolverGpu(int _count) : RK4Solver(_count) {
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

    cudaMalloc( (void**)&mState, mCount * sizeof(double) );
    cudaMalloc( (void**)&mTempStore1, mCount * sizeof(double) );
    cudaMalloc( (void**)&mTempStore2, mCount * sizeof(double) );
    cudaMalloc( (void**)&mTempStore3, mCount * sizeof(double) );
    cudaMalloc( (void**)&mTempStore4, mCount * sizeof(double) );
    cudaMalloc( (void**)&mArg, mCount * sizeof(double) );

    assignArray(mState, 0, mCount);
    assignArray(mTempStore1, 0, mCount);
}

RK4SolverGpu::~RK4SolverGpu() {
    cudaFree(mState);
    cudaFree(mTempStore1);
    cudaFree(mTempStore2);
    cudaFree(mTempStore3);
    cudaFree(mTempStore4);
    cudaFree(mArg);
}

void RK4SolverGpu::copyState(double* result) {
	for (int idx = 0; idx < mCount; ++idx)
		result[idx] = mState[idx];
}

void RK4SolverGpu::prepareArgument(int stage, double timeStep) {

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
