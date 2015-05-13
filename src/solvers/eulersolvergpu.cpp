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
	multipliedArrayByNumber(mTempStore1, timeStep, mTempStore1, mCount);
	sumArray(mState, mTempStore1, mTempStore1, mCount);
}

