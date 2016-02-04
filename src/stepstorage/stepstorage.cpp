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

StepStorage::StepStorage(ProcessingUnit* pu, int count, double _aTol, double _rTol) {
	mCount = count;

	mState = pu->newDoubleArray(mCount);

	aTol = _aTol;
	rTol = _rTol;
}

StepStorage::~StepStorage() {
	// TODO Auto-generated destructor stub
}

void StepStorage::copyState(ProcessingUnit* pu, double* result) {
	pu->copyArray(mState, result, mCount);
}

void StepStorage::saveState(ProcessingUnit* pu, char* path) {
	saveMState(pu, path);
}

void StepStorage::loadState(ProcessingUnit* pu, char* path) {
	loadMState(pu, path);
}

void StepStorage::saveStateWithTempStore(ProcessingUnit* pu, char* path) {
	saveMState(pu, path);
	saveMTempStores(pu, path);
}

void StepStorage::loadStateWithTempStore(ProcessingUnit* pu, char* path) {
	loadMState(pu, path);
	loadMTempStores(pu, path);
}

