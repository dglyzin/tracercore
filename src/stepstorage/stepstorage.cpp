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

void StepStorage::saveState(ProcessingUnit* pu, std::ofstream& out) {
	saveMState(pu, out);
}

void StepStorage::loadState(ProcessingUnit* pu, std::ifstream& in) {
	loadMState(pu, in);
}

void StepStorage::saveStateWithTempStore(ProcessingUnit* pu, std::ofstream& out) {
	saveMState(pu, out);
	saveMTempStores(pu, out);
}

void StepStorage::loadStateWithTempStore(ProcessingUnit* pu, std::ifstream& in) {
	loadMState(pu, in);
	loadMTempStores(pu, in);
}

void StepStorage::saveMState(ProcessingUnit* pu, std::ofstream& out) {
	/*double* toSave = new double [mCount];

	pu->copyArray(mState, toSave, mCount);

	out.write((char*)toSave, SIZE_DOUBLE * mCount);

	delete toSave;*/
	pu->saveArray(mState, mCount, out);
}

void StepStorage::loadMState(ProcessingUnit* pu, std::ifstream& in) {
	/*double* toLoad = new double [mCount];

	in.read((char*)toLoad, SIZE_DOUBLE * mCount);

	pu->copyArray(toLoad, mState, mCount);

	delete toLoad;*/
	pu->loadArray(mState, mCount, in);
}
