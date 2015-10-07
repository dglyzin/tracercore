/*
 * rk4storage.cpp
 *
 *  Created on: 01 окт. 2015 г.
 *      Author: frolov
 */

#include "rk4storage.h"

RK4Storage::RK4Storage() : StepStorage() {
	mTempStore1 = NULL;
	mTempStore2 = NULL;
	mTempStore3 = NULL;
	mTempStore4 = NULL;

	mArg = NULL;
}

RK4Storage::RK4Storage(ProcessingUnit* pu, int count, double _aTol, double _rTol) : StepStorage(pu, count, _aTol, _rTol) {
	mTempStore1 = pu->newDoubleArray(mCount);
	mTempStore2 = pu->newDoubleArray(mCount);
	mTempStore3 = pu->newDoubleArray(mCount);
	mTempStore4 = pu->newDoubleArray(mCount);

	mArg = pu->newDoubleArray(mCount);
}

RK4Storage::~RK4Storage() {
	// TODO Auto-generated destructor stub
}

double* RK4Storage::getStageSource(int stage) {
	switch (stage) {
		case 0:
			return mState;
		case 1: case 2: case 3:
			return mArg;
		default:
			// TODO реально надо assert?
			assert(0);
			return NULL;
	}
}

double* RK4Storage::getStageResult(int stage) {
	switch (stage) {
		case 0:
			return mTempStore1;
		case 1:
			return mTempStore2;
		case 3:
			return mTempStore3;
		case 4:
			return mTempStore4;
		default:
			assert(0);
			return NULL;
	}
}

double RK4Storage::getStageTimeStep(int stage) {
	switch (stage) {
		case 0:
			return 0.0;
		case 1:
			return 0.5;
		case 2:
			return 0.5;
		case 3:
			return 1.0;
		default:
			assert(0);
			return 0.0;
	}
}

void RK4Storage::prepareArgument(ProcessingUnit* pu, int stage, double timestep) {
	switch (stage) {
		case 0:
			pu->multiplyArrayByNumber(mArg, mTempStore1, 0.5*timestep, mCount);
			pu->sumArrays(mArg, mArg, mState, mCount);
			break;
		case 1:
			pu->multiplyArrayByNumber(mArg, mTempStore2, 0.5*timestep, mCount);
			pu->sumArrays(mArg, mArg, mState, mCount);
			break;
		case 2:
			pu->multiplyArrayByNumber(mArg, mTempStore3, 1.0*timestep, mCount);
			pu->sumArrays(mArg, mArg, mState, mCount);
			break;
		case 3:
			/*pc->multiplyArrayByNumber(mTempStore1, mTempStore1, b1, mCount);
			pc->multiplyArrayByNumber(mTempStore2, mTempStore2, b2, mCount);
			pc->multiplyArrayByNumber(mTempStore3, mTempStore3, b3, mCount);
			pc->multiplyArrayByNumber(mTempStore4, mTempStore4, b4, mCount);

			pc->sumArrays(mArg, mTempStore1, mTempStore2, mCount);
			pc->sumArrays(mArg, mArg, mTempStore3, mCount);
			pc->sumArrays(mArg, mArg, mTempStore4, mCount);

			pc->multiplyArrayByNumber(mArg, mArg, timestep, mCount);

			pc->sumArrays(mArg, mArg, mState, mCount);*/

			pu->multiplyArrayByNumber(mArg, mTempStore1, b1, mCount);
			pu->multiplyArrayByNumberAndSum(mArg, mTempStore2, b2, mArg, mCount);
			pu->multiplyArrayByNumberAndSum(mArg, mTempStore3, b3, mArg, mCount);
			pu->multiplyArrayByNumberAndSum(mArg, mTempStore4, b4, mArg, mCount);

			pu->multiplyArrayByNumber(mArg, mArg, timestep, mCount);

			pu->sumArrays(mArg, mState, mArg, mCount);

			break;
		default:
			break;
	}
}

void RK4Storage::confirmStep(ProcessingUnit* pu, double timestep) {
	double* temp = mState;
	mState = mArg;
	mArg = temp;
}

void RK4Storage::rejectStep(ProcessingUnit* pu, double timestep) {
	return;
}

double RK4Storage::getStepError(double timestep) {
	return 0.0;
}

bool RK4Storage::isFSAL() {
	return false;
}

bool RK4Storage::isVariableStep() {
	return false;
}

int RK4Storage::getStageCount() {
	return 4;
}

double RK4Storage::getNewStep(double timestep, double error, int totalDomainElements) {
	return timestep;
}

bool RK4Storage::isErrorPermissible(double error, int totalDomainElements) {
	return true;
}

double* RK4Storage::getDenseOutput(StepStorage* secondState) {
	printf("\nRK4 dense output DON'T WORK!\n");
	return NULL;
}
