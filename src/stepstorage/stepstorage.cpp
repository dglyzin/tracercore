/*
 * stepstorage.cpp
 *
 *  Created on: 24 сент. 2015 г.
 *      Author: frolov
 */

#include "stepstorage.h"

using namespace std;

StepStorage::StepStorage() {
	pu = NULL;
	mCount = 0;
	mState = NULL;
	aTol = 0;
	rTol = 0;
}

StepStorage::StepStorage(ProcessingUnit* _pu, int count, double _aTol, double _rTol) {
	pu = _pu;

	mCount = count;

	mState = pu->newDoubleArray(mCount);

	aTol = _aTol;
	rTol = _rTol;
}

StepStorage::~StepStorage() {
}

void StepStorage::saveMState(char* path) {
	pu->saveArray(mState, mCount, path);
}

void StepStorage::loadMState(std::ifstream& in) {
	pu->loadArray(mState, mCount, in);
}

void StepStorage::copyState(double* result) {
	pu->copyArray(mState, result, mCount);
}

void StepStorage::saveState(char* path) {
	saveMState(path);
}

void StepStorage::loadState(ifstream& in) {
	loadMState(in);
}

void StepStorage::saveStateWithTempStore(char* path) {
	saveMState(path);
	saveMTempStores(path);
}

void StepStorage::loadStateWithTempStore(ifstream& in) {
	loadMState(in);
	loadMTempStores(in);
}

double* StepStorage::getStatePointer() {
	return mState;
}

bool StepStorage::isNan() {
	return pu->isNan(mState, mCount);
}

int StepStorage::getSize(int elementCount) {
	int size = 0;

	size += elementCount * SIZE_DOUBLE;
	size += getSizeChild(elementCount);

	return size;
}
