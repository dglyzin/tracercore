/*
 * ordinary.cpp
 *
 *  Created on: 24 сент. 2015 г.
 *      Author: frolov
 */

#include "ordinary.h"

Ordinary::Ordinary(ProcessingUnit* pu, int solverType, int count, double aTol, double rTol) {
	mStepStorage = createStageStorage(pu, solverType, count, aTol, rTol);

	mSourceStorage = new double* [1];
}

Ordinary::~Ordinary() {
	delete mStepStorage;

	delete mSourceStorage;
}

double** Ordinary::getSource(int stage) {
	mSourceStorage[0] = mStepStorage->getStageSource(stage);

	return mSourceStorage;
}

double* Ordinary::getResult(int stage) {
	return mStepStorage->getStageResult(stage);
}

void Ordinary::prepareArgument(ProcessingUnit* pu, int stage, double timestep) {
	mStepStorage->prepareArgument(pu, stage, timestep);
}

double* Ordinary::getCurrentStateStageData(int stage) {
	return mStepStorage->getStageSource(stage);
}

double Ordinary::getStepError(ProcessingUnit* pu, double timestep) {
	return mStepStorage->getStepError(pu, timestep);
}

void Ordinary::confirmStep(ProcessingUnit* pu, double timestep) {
	mStepStorage->confirmStep(pu, timestep);
}

void Ordinary::rejectStep(ProcessingUnit* pu, double timestep) {
	mStepStorage->rejectStep(pu, timestep);
}

void Ordinary::loadData(ProcessingUnit* pu, double* data) {
	mStepStorage->loadState(pu, data);
}

void Ordinary::getCurrentState(ProcessingUnit* pu, double* result) {
	mStepStorage->copyState(pu, result);
}

double* Ordinary::getCurrentStatePointer() {
	return mStepStorage->getStatePointer();
}

void Ordinary::saveState(ProcessingUnit* pu, char* path) {
	int count;
	double* tmpState = new double [count];
	mStepStorage->copyState();

	out.write((char*), SIZE_DOUBLE * count);
	delete tmpState;
}
