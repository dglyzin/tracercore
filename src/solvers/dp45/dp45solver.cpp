/*
 * dp45solver.cpp
 *
 *  Created on: 29 апр. 2015 г.
 *      Author: frolov
 */

#include "dp45solver.h"

using namespace std;

DP45Solver::DP45Solver() : Solver() {
	mTempStore1 = mTempStore2 = mTempStore3 = mTempStore4 =
			mTempStore5 = mTempStore6 = mTempStore7 = mArg = NULL;
}

DP45Solver::DP45Solver(int _count, double _aTol, double _rTol) : Solver(_count, _aTol, _rTol) {
	mTempStore1 = mTempStore2 = mTempStore3 = mTempStore4 =
			mTempStore5 = mTempStore6 = mTempStore7 = mArg = NULL;
}

DP45Solver::~DP45Solver() {
}

double* DP45Solver::getStageSource(int stage){
	if      (stage == 0) return mArg;
	else if (stage == 1) return mArg;
	else if (stage == 2) return mArg;
	else if (stage == 3) return mArg;
	else if (stage == 4) return mArg;
	else if (stage == 5) return mArg;
	else if (stage == -1) return mState;
	else assert(0);
	return NULL;
}

double* DP45Solver::getStageResult(int stage){
	if      (stage == 0) return mTempStore2;
	else if (stage == 1) return mTempStore3;
	else if (stage == 2) return mTempStore4;
	else if (stage == 3) return mTempStore5;
	else if (stage == 4) return mTempStore6;
	else if (stage == 5) return mTempStore7;
	else if (stage == -1) return mTempStore1;
	else assert(0);
	return NULL;
}

double DP45Solver::getStageTimeStep(int stage){
    if      (stage == 0) return c2;
    else if (stage == 1) return c3;
    else if (stage == 2) return c4;
    else if (stage == 3) return c5;
    else if (stage == 4) return 1.0;
    else if (stage == 5) return 1.0;
    else if (stage ==-1) return 0.0;
    else assert(0);
    return 0.0;
}

void DP45Solver::confirmStep(double timestep){
    double* temp = mState;
    mState = mArg;
    mArg = temp;

    temp = mTempStore7;
    mTempStore7 = mTempStore1;
    mTempStore1 = temp;

    prepareFSAL(timestep);
/*#pragma omp parallel for
	for (int idx=0; idx<mCount; idx++)
		mArg[idx] = mState[idx]+a21*timestep*mTempStore7[idx];*/
}


void DP45Solver::rejectStep(double timestep){
    prepareFSAL(timestep);
}


double DP45Solver::getNewStep(double timeStep, double error, int totalDomainElements){
	double err = sqrt(error/totalDomainElements);
	return timeStep * min( facmax, max( facmin, fac * pow(1.0/err,1.0/5.0) ) );
	//return timeStep;
}

bool DP45Solver::isErrorPermissible(double error, int totalDomainElements){
	double err = sqrt(error/totalDomainElements);
	if (err < 1)
		return true;
	else
		return false;
	//return true;
}
