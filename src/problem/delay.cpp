/*
 * delay.cpp
 *
 *  Created on: 16 дек. 2015 г.
 *      Author: frolov
 */

#include "delay.h"

Delay::Delay(ProcessingUnit* pu, int solverType, int count, double aTol, double rTol, int _delayCount) {
	//TODO fix!!! реализация функции для ProcessingUnit!!!
	maxStorageCount = 1;

	mStepStorage = new StepStorage* [maxStorageCount];

	for (int i = 0; i < maxStorageCount; ++i) {
		mStepStorage[i] = createStageStorage(pu, solverType, count, aTol, rTol);
	}

	delayCount = _delayCount;

	mSourceStorage = new double* [delayCount + 1];
}

Delay::~Delay() {
	for (int i = 0; i < maxStorageCount; ++i) {
		delete mStepStorage[i];
	}

	delete mStepStorage;

	delete mSourceStorage;
}

double** Delay::getSource(int stage, double time) {
	int sourceStorageNumber = getSourceStorageNumber(time);
	mSourceStorage[0] = mStepStorage[sourceStorageNumber]->getStageSource(stage);

	for (int i = 0; i < delayCount; ++i) {
		//mSourceStorage[i + 1] = mStepStorage[3]->getDenseOutput()
	}

	return mSourceStorage;
}

double* Delay::getResult(int stage, double time) {
	int resultStorageNumber = getResultStorageNumber(time);
	return mStepStorage[resultStorageNumber]->getStageResult(stage);
}

void Delay::prepareArgument(ProcessingUnit* pu, int stage, double timestep) {
	mStepStorage->prepareArgument(pu, stage, timestep);
}

double* Delay::getCurrentStateStageData(int stage) {
	return mStepStorage->getStageSource(stage);
}

double Delay::getStepError(ProcessingUnit* pu, double timestep) {
	return mStepStorage->getStepError(pu, timestep);
}

void Delay::confirmStep(ProcessingUnit* pu, double timestep) {
	mStepStorage->confirmStep(pu, timestep);
}

void Delay::rejectStep(ProcessingUnit* pu, double timestep) {
	mStepStorage->rejectStep(pu, timestep);
}

void Delay::loadData(ProcessingUnit* pu, double* data) {
	mStepStorage->loadState(pu, data);
}

void Delay::getCurrentState(ProcessingUnit* pu, double* result) {
	mStepStorage->copyState(pu, result);
}

double* Delay::getCurrentStatePointer() {
	return mStepStorage->getStatePointer();
}

int Delay::getSourceStorageNumber(double time) {
	printf("\nget source storage number don't work!!!\n");
	return 0;
}

int Delay::getResultStorageNumber(double time) {
	printf("\nget result storage number don't work!!!\n");
	return 0;
}
