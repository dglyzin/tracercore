/*
 * eulersolvercpu.cpp
 *
 *  Created on: 28 апр. 2015 г.
 *      Author: frolov
 */

#include "eulersolvercpu.h"

EulerSolverCpu::EulerSolverCpu(int _count) : EulerSolver(_count) {
	mState = new double [mCount];
	mTempStore1 = new double [mCount];
}

EulerSolverCpu::~EulerSolverCpu() {
	delete mState;
	delete mTempStore1;
}


void EulerSolverCpu::copyState(double* result) {
	for (int idx = 0; idx < mCount; ++idx)
		result[idx] = mState[idx];
}

void EulerSolverCpu::prepareArgument(int stage, double timeStep) {
#pragma omp parallel for
	for (int idx = 0; idx < mCount; ++idx)
	    mTempStore1[idx]= mState[idx] + timeStep * mTempStore1[idx];
}
