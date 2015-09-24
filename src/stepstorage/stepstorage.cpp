/*
 * stepstorage.cpp
 *
 *  Created on: 24 сент. 2015 г.
 *      Author: frolov
 */

#include "stepstorage.h"

StepStorage::StepStorage() {
	mCount = 0;
	mState = NULL;
	aTol = 0;
	rTol = 0;
}

StepStorage::StepStorage(ProcessingUnit* pc, int count, double _aTol, double _rTol) {
	mCount = count;

	mState = pc->newDoubleArray(mCount);

	aTol = _aTol;
	rTol = _rTol;
}

StepStorage::~StepStorage() {
	// TODO Auto-generated destructor stub
}

void StepStorage::copyState(ProcessingUnit* pc, double* result) {
	pc->copyArray(mState, result, mCount);
}

void StepStorage::loadState(ProcessingUnit* pc, double* data) {
	pc->copyArray(data, mState, mCount);
}
