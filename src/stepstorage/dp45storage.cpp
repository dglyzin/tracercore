/*
 * dp45storage.cpp
 *
 *  Created on: 06 окт. 2015 г.
 *      Author: frolov
 */

#include "dp45storage.h"

DP45Storage::DP45Storage() : StepStorage() {
	mTempStore1 = NULL;
	mTempStore2 = NULL;
	mTempStore3 = NULL;
	mTempStore4 = NULL;
	mTempStore5 = NULL;
	mTempStore6 = NULL;
	mTempStore7 = NULL;

	mArg = NULL;

}

DP45Storage::DP45Storage(ProcessingUnit* pu, int count, double _aTol, double _rTol) : StepStorage(pu, count, _aTol, _rTol) {
	mTempStore1 = pu->newDoubleArray(mCount);
	mTempStore2 = pu->newDoubleArray(mCount);
	mTempStore3 = pu->newDoubleArray(mCount);
	mTempStore4 = pu->newDoubleArray(mCount);
	mTempStore5 = pu->newDoubleArray(mCount);
	mTempStore6 = pu->newDoubleArray(mCount);
	mTempStore7 = pu->newDoubleArray(mCount);

	mArg = pu->newDoubleArray(mCount);
}

DP45Storage::~DP45Storage() {
	// TODO Auto-generated destructor stub
}

double* DP45Storage::getStageSource(int stage) {
	/*if      (stage == 0) return mArg;
	else if (stage == 1) return mArg;
	else if (stage == 2) return mArg;
	else if (stage == 3) return mArg;
	else if (stage == 4) return mArg;
	else if (stage == 5) return mArg;
	else if (stage == -1) return mState;
	else assert(0);
	return NULL;*/

	switch (stage) {
		case 0: case 1: case 2: case 3: case 4: case 5:
			return mArg;
		case -1:
			return mState;
		default:
			assert(0);
			return NULL;
	}
}

double* DP45Storage::getStageResult(int stage) {
	/*if      (stage == 0) return mTempStore2;
	else if (stage == 1) return mTempStore3;
	else if (stage == 2) return mTempStore4;
	else if (stage == 3) return mTempStore5;
	else if (stage == 4) return mTempStore6;
	else if (stage == 5) return mTempStore7;
	else if (stage == -1) return mTempStore1;
	else assert(0);
	return NULL;*/

	switch (stage) {
		case 0:
			return mTempStore2;
		case 1:
			return mTempStore3;
		case 2:
			return mTempStore4;
		case 3:
			return mTempStore5;
		case 4:
			return mTempStore6;
		case 5:
			return mTempStore7;
		case -1:
			return mTempStore1;
		default:
			assert(0);
			return NULL;
	}
}

double DP45Storage::getStageTimeStep(int stage) {
	/*if      (stage == 0) return c2;
	else if (stage == 1) return c3;
	else if (stage == 2) return c4;
	else if (stage == 3) return c5;
	else if (stage == 4) return 1.0;
	else if (stage == 5) return 1.0;
	else if (stage ==-1) return 0.0;
	else assert(0);
	return 0.0;*/

	switch (stage) {
		case 0:
			return c2;
		case 1:
			return c3;
		case 2:
			return c4;
		case 3:
			return c5;
		case 4:
			return 1.0;
		case 5:
			return 1.0;
		case -1:
			return 0.0;
		default:
			assert(0);
			return 0.0;
	}
}

void DP45Storage::prepareArgument(ProcessingUnit* pc, int stage, double timestep) {

}
