/*
 * dp45solvercpu.cpp
 *
 *  Created on: 30 апр. 2015 г.
 *      Author: frolov
 */

#include "dp45solvercpu.h"

using namespace std;

DP45SolverCpu::DP45SolverCpu(int _count, double _aTol, double _rTol) : DP45Solver(_count, _aTol, _rTol) {
	mState = new double [mCount];

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

        mTempStore1[idx] = 1;
        mTempStore2[idx] = 2;
        mTempStore3[idx] = 3;
        mTempStore4[idx] = 4;
        mTempStore5[idx] = 5;
        mTempStore6[idx] = 6;
        mTempStore7[idx] = 7;

        mArg[idx] = 0;
    }
}

DP45SolverCpu::~DP45SolverCpu() {
    delete mState;

    delete mTempStore1;
    delete mTempStore2;
    delete mTempStore3;
    delete mTempStore4;
    delete mTempStore5;
    delete mTempStore6;
    delete mTempStore7;

    delete mArg;
}

void DP45SolverCpu::prepareFSAL(double timeStep) { //must be NEW timestep
#pragma omp parallel for
	for (int idx = 0; idx < mCount; idx++)
		mArg[idx] = mState[idx] + a21 * timeStep * mTempStore1[idx];
}

void DP45SolverCpu::copyState(double* result) {
	for (int idx = 0; idx < mCount; ++idx)
		result[idx] = mState[idx];
}

void DP45SolverCpu::prepareArgument(int stage, double timeStep) {

	if      (stage == 0)
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
			mArg[idx] = mState[idx]+timeStep*(a71*mTempStore1[idx] + a73*mTempStore3[idx] + a74*mTempStore4[idx] + a75*mTempStore5[idx] +a76*mTempStore6[idx]);
	else if (stage == 5)
	{ //nothing to be done here before step is confirmed, moved to confirmStep
	}
	else if (stage == -1)
#pragma omp parallel for
		for (int idx=0; idx<mCount; idx++)
			mArg[idx] = mState[idx]+a21*timeStep*mTempStore1[idx];

	else assert(0);
}

double DP45SolverCpu::getStepError(double timeStep){
	double err=0;
#pragma omp parallel for reduction (+:err)
	for (int idx=0; idx<mCount; idx++){
		double erri =  timeStep * (e1 * mTempStore1[idx] + e3 * mTempStore3[idx] + e4 * mTempStore4[idx] +
	                            e5 * mTempStore5[idx] + e6 * mTempStore6[idx]+ e7 * mTempStore7[idx])
	                          /(aTol/* + rTol * max(mArg[idx], mState[idx])*/);
	   err += erri * erri;
	}

	return err;
}


void DP45SolverCpu::print(int zCount, int yCount, int xCount, int cellSize) {
	printf("mState: \n");
	printMatrix(mState, zCount, yCount, xCount, cellSize);

	printf("mTempStore1: \n");
	printMatrix(mTempStore1, zCount, yCount, xCount, cellSize);

	printf("mTempStore2: \n");
	printMatrix(mTempStore2, zCount, yCount, xCount, cellSize);

	printf("mTempStore3: \n");
	printMatrix(mTempStore3, zCount, yCount, xCount, cellSize);

	printf("mTempStore4: \n");
	printMatrix(mTempStore4, zCount, yCount, xCount, cellSize);

	printf("mTempStore5: \n");
	printMatrix(mTempStore5, zCount, yCount, xCount, cellSize);

	printf("mTempStore6: \n");
	printMatrix(mTempStore6, zCount, yCount, xCount, cellSize);

	printf("mTempStore7: \n");
	printMatrix(mTempStore7, zCount, yCount, xCount, cellSize);

	printf("mArg: \n");
	printMatrix(mArg, zCount, yCount, xCount, cellSize);
}
