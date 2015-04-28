/*
 * solver.cpp
 *
 *  Created on: Feb 12, 2015
 *      Author: dglyzin
 */

#include "solver.h"

/*Solver* GetCpuSolver(int solverIdx, int count){
	if      (solverIdx == EULER)
		return new EulerSolver(count);
	else if (solverIdx == RK4)
		return new EulerSolver(count);
	else
		return new EulerSolver(count);
}

Solver* GetGpuSolver(int solverIdx, int count){
	return NULL;
}*/

Solver::Solver(){
	mCount = 0;
	mState = NULL;
}

SolverInfo::SolverInfo(){

}

void Solver::copyState(double* result){
	for (int idx=0;idx<mCount;++idx)
		result[idx] = mState[idx];
}

//****************1. EULER SOLVER*************//
/*EulerSolver::EulerSolver(int _count){
    mCount = _count;
    mState = new double[mCount];
    mTempStore1 = new double[mCount];
    for (int idx = 0; idx < mCount; ++idx){
        mState[idx] = 0;
        mTempStore1[idx] = 0;
    }
}

EulerSolver::~EulerSolver(){
    delete mState;
    delete mTempStore1;
}

double* EulerSolver::getStageSource(int stage){
    assert(stage == 0);
    return mState;
}

double* EulerSolver::getStageResult(int stage){
    assert(stage == 0);
    return mTempStore1;
}

void EulerSolver::prepareArgument(int stage, double timeStep){
#pragma omp parallel for
	for (int idx = 0; idx < mCount; ++idx)
	    mTempStore1[idx]= mState[idx] + timeStep*mTempStore1[idx];
}

void EulerSolver::confirmStep(double timestep){
    double* temp = mState;
    mState = mTempStore1;
    mTempStore1 = temp;
}*/

//****************2. RK4 SOLVER*************//
/*
 * Stage 0: mState -f-> ts1, mState+h/2 ts1 --> arg
 * Stage 1: arg -f-> ts2, mState+h/2 ts2 --> arg
 * Stage 2: arg -f-> ts3, mState+h ts3   --> arg
 * Stage 3: arg -f-> ts4, mState+h/6 ts1 + h/3 ts2 + h/3 ts3 + h/4 ts4 --> arg
 *
 * confirm: mState<-->arg
 */

RK4Solver::RK4Solver(int _count){
    mCount = _count;
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

RK4Solver::~RK4Solver(){
    delete mState;
    delete mTempStore1;
    delete mTempStore2;
    delete mTempStore3;
    delete mTempStore4;
    delete mArg;
}

void RK4Solver::prepareArgument(int stage, double timeStep) {

	if      (stage == 0)
#pragma omp parallel for
		for (int idx=0; idx<mCount; idx++)
			mArg[idx] = mState[idx]+0.5*timeStep*mTempStore1[idx];
	else if (stage == 1)
#pragma omp parallel for
		for (int idx=0; idx<mCount; idx++)
			mArg[idx] = mState[idx]+0.5*timeStep*mTempStore2[idx];
	else if (stage == 2)
#pragma omp parallel for
		for (int idx=0; idx<mCount; idx++)
			mArg[idx] = mState[idx]+timeStep*mTempStore3[idx];
	else if (stage == 3){
	    const double b1 = 1.0/6.0;
	    const double b2 = 1.0/3.0;
	    const double b3 = 1.0/3.0;
	    const double b4= 1.0/6.0;
	#pragma omp parallel for
		for (int idx=0; idx<mCount; idx++)
			mArg[idx] = mState[idx]+timeStep*(b1*mTempStore1[idx]+b2*mTempStore2[idx]+b3*mTempStore3[idx]+b4*mTempStore4[idx]);
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

//****************3. DP45 SOLVER*************//
const double c2=0.2, c3=0.3, c4=0.8, c5=8.0/9.0;

const double a21=0.2, a31=3.0/40.0, a32=9.0/40.0;
const double a41=44.0/45.0, a42=-56.0/15.0, a43=32.0/9.0;
const double a51=19372.0/6561.0, a52=-25360.0/2187.0;
const double a53=64448.0/6561.0, a54=-212.0/729.0;
const double a61=9017.0/3168.0, a62=-355.0/33.0, a63=46732.0/5247.0;
const double a64=49.0/176.0, a65=-5103.0/18656.0;
const double a71=35.0/384.0, a73=500.0/1113.0, a74=125.0/192.0;
const double a75=-2187.0/6784.0, a76=11.0/84.0;
const double e1=71.0/57600.0, e3=-71.0/16695.0, e4=71.0/1920.0;
const double e5=-17253.0/339200.0, e6=22.0/525.0, e7=-1.0/40.0;
const double facmin=0.5, facmax = 2, fac = 0.9;

/*
 * Stage -1: mState -f-> ts1, mState+a21*ts1 --> arg
 *
 * Stage  0: arg -f-> ts2, mState+(h*a31*ts1+a32*ts2) --> arg
 * Stage  1: arg -f-> ts3, mState+(h*a41*ts1+a42*ts2+a43*ts3) --> arg
 * Stage  2-3:..
 * Stage  4: arg -f-> ts6, mState+...   --> arg (possible next state)
 * Stage  5: arg -f-> ts7
 *
 * error:  +e_i*ts_i
 *
 * confirm: mState<-->arg,  mState+a21*ts7 --> arg
 */

void DP45Solver::prepareArgument(int stage, double timeStep) {

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
			mArg[idx] = mState[idx]+timeStep*(a61*mTempStore1[idx] + a62*mTempStore2[idx] + a63*mTempStore3[idx] + a64*mTempStore4[idx] +a65*mTempStore5[idx]);
	else if (stage == 5)
	{ //nothing to be done here before step is confirmed, moved to confirmStep
	}
	else if (stage == -1)
#pragma omp parallel for
		for (int idx=0; idx<mCount; idx++)
			mArg[idx] = mState[idx]+a21*timeStep*mTempStore1[idx];

	else assert(0);
}


DP45Solver::DP45Solver(int _count){
    mCount = _count;
    mState = new double[mCount];
    mTempStore1 = new double[mCount];
    mTempStore2 = new double[mCount];
    mTempStore3 = new double[mCount];
    mTempStore4 = new double[mCount];
    mTempStore5 = new double[mCount];
    mTempStore6 = new double[mCount];
    mTempStore7 = new double[mCount];
    mArg = new double[mCount];
    for (int idx = 0; idx < mCount; ++idx){
        mState[idx] = 0;
        mTempStore1[idx] = 0;
    }
}

DP45Solver::~DP45Solver(){
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

static double max_d (double a, double b){
  return (a > b)?a:b;
} /* max_d */
static double min_d (double a, double b){
  return (a < b)?a:b;
} /* min_d */


double DP45Solver::getStepError(double timeStep, double aTol, double rTol){
	double err=0;
#pragma omp parallel for reduction (+:err)
	for (int idx=0; idx<mCount; idx++){
		double erri = timeStep*(e1*mTempStore1[idx] + e3*mTempStore3[idx] + e4*mTempStore4[idx] +
	                            e5*mTempStore5[idx] + e6*mTempStore6[idx]+ e7*mTempStore7[idx])
	                          /(aTol+rTol*max_d(mArg[idx], mState[idx]));
	   err+=erri*erri;
	}

	return err;
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

void DP45Solver::confirmStep(double timestep){
    double* temp = mState;
    mState = mArg;
    mArg = temp;
#pragma omp parallel for
	for (int idx=0; idx<mCount; idx++)
		mArg[idx] = mState[idx]+a21*timestep*mTempStore7[idx];
}



double DP45SolverInfo::getNewStep(double timeStep, double error, int totalDomainElements){
	double err = sqrt(error/totalDomainElements);
	return timeStep * min_d(facmax,max_d(facmin,fac*pow(1.0/err,1.0/5.0)));
}
int DP45SolverInfo::isErrorOK(double error, int totalDomainElements){
	double err = sqrt(error/totalDomainElements);
	if (err<1)
		return 1;
	else
		return 0;
}
