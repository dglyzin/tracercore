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
		int sourceStorageNumberDelay = getSourceStorageNumberDelay(time, i);
		//mSourceStorage[i + 1] mStepStorage[sourceStorageNumberDelay]->getDenseOutput();
	}

	return mSourceStorage;
}

double* Delay::getResult(int stage, double time) {
	int resultStorageNumber = getResultStorageNumber(time);
	return mStepStorage[resultStorageNumber]->getStageResult(stage);
}

void Delay::prepareArgument(ProcessingUnit* pu, int stage, double timestep) {
	int currentStorageNumber = getCurrentStorageNumber();
	mStepStorage[currentStorageNumber]->prepareArgument(pu, stage, timestep);
}

double* Delay::getCurrentStateStageData(int stage) {
	int currentStorageNumber = getCurrentStorageNumber();
	return mStepStorage[currentStorageNumber]->getStageSource(stage);
}

double Delay::getStepError(ProcessingUnit* pu, double timestep) {
	int currentStorageNumber = getCurrentStorageNumber();
	return mStepStorage[currentStorageNumber]->getStepError(pu, timestep);
}

void Delay::confirmStep(ProcessingUnit* pu, double timestep) {
	int currentStorageNumber = getCurrentStorageNumber();
	mStepStorage[currentStorageNumber]->confirmStep(pu, timestep);
}

void Delay::rejectStep(ProcessingUnit* pu, double timestep) {
	int currentStorageNumber = getCurrentStorageNumber();
	mStepStorage[currentStorageNumber]->rejectStep(pu, timestep);
}

void Delay::loadData(ProcessingUnit* pu, double* data) {
	int currentStorageNumber = getCurrentStorageNumber();
	mStepStorage[currentStorageNumber]->loadState(pu, data);
}

void Delay::getCurrentState(ProcessingUnit* pu, double* result) {
	int currentStorageNumber = getCurrentStorageNumber();
	mStepStorage[currentStorageNumber]->copyState(pu, result);
}

double* Delay::getCurrentStatePointer() {
	int currentStorageNumber = getCurrentStorageNumber();
	return mStepStorage[currentStorageNumber]->getStatePointer();
}

int Delay::getSourceStorageNumber(double time) {
	printf("\nget source storage number don't work!!!\n");
	return 0;
}

int Delay::getSourceStorageNumberDelay(double time, int delayNumber) {
	printf("\nget source storage number for delay don't work!!!\n");
	return 0;
}

int Delay::getResultStorageNumber(double time) {
	printf("\nget result storage number don't work!!!\n");
	return 0;
}

int Delay::getCurrentStorageNumber() {
	printf("\nget current storage number don't work!!!\n");
	return 0;
}
