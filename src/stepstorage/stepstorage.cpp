/*
 * stepstorage.cpp
 *
 *  Created on: 24 сент. 2015 г.
 *      Author: frolov
 */

#include "stepstorage.h"

StepStorage::StepStorage(ProcessingUnit* pc, int count, double _aTol, double _rTol) {
	mCount = count;

	mState = pc->newDoubleArray(mCount);

	aTol = _aTol;
	rTol = _rTol;
}

StepStorage::~StepStorage() {
	// TODO Auto-generated destructor stub
}

