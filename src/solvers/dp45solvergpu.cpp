/*
 * dp45solvergpu.cpp
 *
 *  Created on: 14 мая 2015 г.
 *      Author: frolov
 */

#include "dp45solvergpu.h"

using namespace std;

DP45SolverGpu::DP45SolverGpu(int _count) : DP45Solver(_count) {
	/*mState = new double [mCount];

	mTempStore1 = new double [mCount];
	mTempStore2 = new double [mCount];
	mTempStore3 = new double [mCount];
	mTempStore4 = new double [mCount];
	mTempStore5 = new double [mCount];
	mTempStore6 = new double [mCount];
	mTempStore7 = new double [mCount];

	mArg = new double [mCount];

    for (int idx = 0; idx < mCount; ++idx){
        mState[idx] = 0;
        mTempStore1[idx] = 0;
    }*/

    cudaMalloc( (void**)&mState, mCount * sizeof(double) );
    cudaMalloc( (void**)&mTempStore1, mCount * sizeof(double) );
    cudaMalloc( (void**)&mTempStore2, mCount * sizeof(double) );
    cudaMalloc( (void**)&mTempStore3, mCount * sizeof(double) );
    cudaMalloc( (void**)&mTempStore4, mCount * sizeof(double) );
    cudaMalloc( (void**)&mTempStore5, mCount * sizeof(double) );
    cudaMalloc( (void**)&mTempStore6, mCount * sizeof(double) );
    cudaMalloc( (void**)&mTempStore7, mCount * sizeof(double) );
    cudaMalloc( (void**)&mArg, mCount * sizeof(double) );

    assignArray(mState, 0, mCount);
    assignArray(mTempStore1, 0, mCount);
    assignArray(mTempStore2, 0, mCount);
    assignArray(mTempStore3, 0, mCount);
    assignArray(mTempStore4, 0, mCount);
    assignArray(mTempStore5, 0, mCount);
    assignArray(mTempStore6, 0, mCount);
    assignArray(mTempStore7, 0, mCount);
    assignArray(mArg, 0, mCount);
}

DP45SolverGpu::~DP45SolverGpu() {
    cudaFree(mState);
    cudaFree(mTempStore1);
    cudaFree(mTempStore2);
    cudaFree(mTempStore3);
    cudaFree(mTempStore4);
    cudaFree(mTempStore5);
    cudaFree(mTempStore6);
    cudaFree(mTempStore7);
    cudaFree(mArg);
}

void DP45SolverGpu::prepareFSAL(double timeStep) {
/*#pragma omp parallel for
	for (int idx = 0; idx < mCount; idx++)
		mArg[idx] = mState[idx] + a21 * timeStep * mTempStore7[idx];*/
	multipliedArrayByNumber(mTempStore7, a21 * timeStep, mArg, mCount);
	sumArray(mArg, mState, mArg, mCount);
}

void DP45SolverGpu::copyState(double* result) {
	/*for (int idx = 0; idx < mCount; ++idx)
		result[idx] = mState[idx];*/
	cudaMemcpy(result, mState, mCount * sizeof(double), cudaMemcpyDeviceToHost);
}

void DP45SolverGpu::prepareArgument(int stage, double timeStep) {

	/*if      (stage == 0)
#pragma omp parallel for
		for (int idx=0; idx<mCount; idx++)
			mArg[idx] = mState[idx]+timeStep*(a31*mTempStore1[idx] + a32*mTempStore2[idx]);
	else if (stage == 1)
#pragma omp parallel for
		for (int idx=0; idx<mCount; idx++)
			mArg[idx] = mState[idx]+timeStep*(a41*mTempStore1[idx] + a42*mTempStore2[idx] + a43*mTempStore3[idx]);
	else if (stage == 2)
#pragma omp parallel for
		for (int idx=0; idx<mCount; idx++)
			mArg[idx] = mState[idx]+timeStep*(a51*mTempStore1[idx] + a52*mTempStore2[idx] + a53*mTempStore3[idx] + a54*mTempStore4[idx]);
	else if (stage == 3)
#pragma omp parallel for
		for (int idx=0; idx<mCount; idx++)
			mArg[idx] = mState[idx]+timeStep*(a61*mTempStore1[idx] + a62*mTempStore2[idx] + a63*mTempStore3[idx] + a64*mTempStore4[idx] +a65*mTempStore5[idx]);
	else if (stage == 4)
#pragma omp parallel for
		for (int idx=0; idx<mCount; idx++)
			mArg[idx] = mState[idx]+timeStep*(a61*mTempStore1[idx] + a62*mTempStore2[idx] + a63*mTempStore3[idx] + a64*mTempStore4[idx] +a65*mTempStore5[idx]);
	else if (stage == 5)
	{ //nothing to be done here before step is confirmed, moved to confirmStep
	}
	else if (stage == -1)
#pragma omp parallel for
		for (int idx=0; idx<mCount; idx++)
			mArg[idx] = mState[idx]+a21*timeStep*mTempStore1[idx];

	else assert(0);*/

	switch (stage) {
		case 0:
			multipliedByNumberAndSumArrays(mTempStore1, a31, mTempStore2, a32, mArg, mCount);
			multipliedArrayByNumber(mArg, timeStep, mArg, mCount);
			sumArray(mArg, mState, mArg, mCount);
			break;

		case 1:
			multipliedByNumberAndSumArrays(mTempStore1, a41, mTempStore2, a42, mTempStore3, a43, mArg, mCount);
			multipliedArrayByNumber(mArg, timeStep, mArg, mCount);
			sumArray(mArg, mState, mArg, mCount);
			break;

		case 2:
			multipliedByNumberAndSumArrays(mTempStore1, a51, mTempStore2, a52, mTempStore3, a53, mTempStore4, a54, mArg, mCount);
			multipliedArrayByNumber(mArg, timeStep, mArg, mCount);
			sumArray(mArg, mState, mArg, mCount);
			break;

			// TODO check!!
		case 3: case 4:
			multipliedByNumberAndSumArrays(mTempStore1, a61, mTempStore2, a62, mTempStore3, a63, mTempStore4, a64, mTempStore5, a65, mArg, mCount);
			multipliedArrayByNumber(mArg, timeStep, mArg, mCount);
			sumArray(mArg, mState, mArg, mCount);
			break;

		case 5:
			break;

		case -1:
			multipliedArrayByNumber(mTempStore1, a21 * timeStep, mArg, mCount);
			sumArray(mArg, mState, mArg, mCount);
			break;

		default:
			assert(0);
			break;
	}
}

double DP45SolverGpu::getStepError(double timeStep, double aTol, double rTol){
	double err=0;
#pragma omp parallel for reduction (+:err)
	for (int idx=0; idx<mCount; idx++){
		double erri = timeStep * (e1 * mTempStore1[idx] + e3 * mTempStore3[idx] + e4 * mTempStore4[idx] +
	                            e5 * mTempStore5[idx] + e6 * mTempStore6[idx]+ e7 * mTempStore7[idx])
	                          /(aTol + rTol * max(mArg[idx], mState[idx]));
	   err += erri * erri;
	}

	return err;
}

