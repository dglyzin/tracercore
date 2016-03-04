/*
 * stepstorage.cpp
 *
 *  Created on: 24 сент. 2015 г.
 *      Author: frolov
 */

#include "stepstorage.h"

using namespace std;

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
}

void StepStorage::saveMState(ProcessingUnit* pu, char* path) {
	pu->saveArray(mState, mCount, path);
}

void StepStorage::loadMState(ProcessingUnit* pu, std::ifstream& in) {
	pu->loadArray(mState, mCount, in);
}

void StepStorage::copyState(ProcessingUnit* pu, double* result) {
	pu->copyArray(mState, result, mCount);
}

void StepStorage::saveState(ProcessingUnit* pu, char* path) {
	saveMState(pu, path);
}

void StepStorage::loadState(ProcessingUnit* pu, ifstream& in) {
	loadMState(pu, in);
}

void StepStorage::saveStateWithTempStore(ProcessingUnit* pu, char* path) {
	saveMState(pu, path);
	saveMTempStores(pu, path);
}

void StepStorage::loadStateWithTempStore(ProcessingUnit* pu, ifstream& in) {
	loadMState(pu, in);
	loadMTempStores(pu, in);
}

double* StepStorage::getStatePointer() {
	return mState;
}

bool StepStorage::isNan(ProcessingUnit* pu) {
	return pu->isNan(mState, mCount);
}
