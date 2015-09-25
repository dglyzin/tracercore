/*
 * eulerstorage.cpp
 *
 *  Created on: 24 сент. 2015 г.
 *      Author: frolov
 */

#include "eulerstorage.h"

EulerStorage::EulerStorage() : StepStorage() {
	mTempStore1 = NULL;
}

EulerStorage::EulerStorage(ProcessingUnit* pu, int count, double _aTol, double _rTol) : StepStorage(pu, count, _aTol, _rTol) {
	mTempStore1 = pu->newDoubleArray(mCount);
}

EulerStorage::~EulerStorage() {
	// TODO Auto-generated destructor stub
}

double* EulerStorage::getStageSource(int stahe) {
	return mState;
}

double* EulerStorage::getStageResult(int stage) {
	return mTempStore1;
}

double EulerStorage::getStageTimeStep(int stage) {
	return 0.0;
}

void EulerStorage::prepareArgument(ProcessingUnit* pc, int stage, double timestep) {
	pc->multiplyArrayByNumber(mTempStore1, mTempStore1, timestep, mCount);
	pc->sumArrays(mTempStore1, mState, mTempStore1, mCount);
}

void EulerStorage::confirmStep(double timestep) {
    double* temp = mState;
    mState = mTempStore1;
    mTempStore1 = temp;
}

void EulerStorage::rejectStep(double timestep) {
	return;
}

double EulerStorage::getStepError(double timestep) {
	return 0.0;
}

bool EulerStorage::isFSAL() {
	return false;
}

bool EulerStorage::isVariableStep() {
	return false;
}

int EulerStorage::getStageCount() {
	return 1;
}

double EulerStorage::getNewStep(double timeStep, double error, int totalDomainElements) {
	return 0.0;
}

bool EulerStorage::isErrorPermissible(double error, int totalDomainElements) {
	return true;
}

double* EulerStorage::getDenseOutput(StepStorage* secondState) {
	printf("\nEuler dense output DON'T WORK!\n");
	return NULL;
}
