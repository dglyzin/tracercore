/*
 * eulersolvergpu.cpp
 *
 *  Created on: 08 мая 2015 г.
 *      Author: frolov
 */

#include "eulersolvergpu.h"

EulerSolverGpu::EulerSolverGpu(int _count, double _aTol, double _rTol) : EulerSolver(_count, _aTol, _rTol) {
	/*mState = new double [mCount];
	mTempStore1 = new double [mCount];*/
	cudaMalloc( (void**)&mState, mCount * sizeof(double) );
	cudaMalloc( (void**)&mTempStore1, mCount * sizeof(double) );
}

EulerSolverGpu::~EulerSolverGpu() {
	cudaFree(mState);
	cudaFree(mTempStore1);
}


void EulerSolverGpu::copyState(double* result) {
	cudaMemcpy(result, mState, mCount * sizeof(double), cudaMemcpyDeviceToHost);
}

void EulerSolverCpu::loadState(double* data) {
	cudaMemcpy(mState, data, mCount * sizeof(double), cudaMemcpyHostToDevice);
}

void EulerSolverGpu::prepareArgument(int stage, double timeStep) {
	multiplyArrayByNumber(mTempStore1, timeStep, mTempStore1, mCount);
	sumArray(mState, mTempStore1, mTempStore1, mCount);
}

