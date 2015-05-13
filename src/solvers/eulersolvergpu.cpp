/*
 * eulersolvergpu.cpp
 *
 *  Created on: 08 мая 2015 г.
 *      Author: frolov
 */

#include "eulersolvergpu.h"

EulerSolverGpu::EulerSolverGpu(int _count) : EulerSolver(_count) {
	/*mState = new double [mCount];
	mTempStore1 = new double [mCount];*/
	cudaMalloc( (void**)&mState, mCount );
	cudaMalloc( (void**)&mTempStore1, mCount );
}

EulerSolverGpu::~EulerSolverGpu() {
	cudaFree(mState);
	cudaFree(mTempStore1);
}


void EulerSolverGpu::copyState(double* result) {
	cudaMemcpy(result, mState, mCount, cudaMemcpyDeviceToHost);
}

void EulerSolverGpu::prepareArgument(int stage, double timeStep) {
/*#pragma omp parallel for
	for (int idx = 0; idx < mCount; ++idx)
	    mTempStore1[idx]= mState[idx] + timeStep * mTempStore1[idx];*/
	multipliedArrayByNumber(mTempStore1, timeStep, mCount);
	sumArray(mState, mTempStore1, mTempStore1, mCount);
}

