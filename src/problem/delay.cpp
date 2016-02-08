/*
 * delay.cpp
 *
 *  Created on: 16 дек. 2015 г.
 *      Author: frolov
 */

#include "delay.h"

using namespace std;

Delay::Delay(ProcessingUnit* pu, int solverType, int count, double aTol, double rTol, int _delayCount) {
	//TODO fix!!! реализация функции для ProcessingUnit!!!
	maxStorageCount = 1;

	mStepStorage = new StepStorage* [maxStorageCount];

	for (int i = 0; i < maxStorageCount; ++i) {
		mStepStorage[i] = createStageStorage(pu, solverType, count, aTol, rTol);
	}

	delayCount = _delayCount;

	mSourceStorage = new double* [delayCount + 1];

	currentStorageNumber = 0;
}

Delay::~Delay() {
	for (int i = 0; i < maxStorageCount; ++i) {
		delete mStepStorage[i];
	}

	delete mStepStorage;

	delete mSourceStorage;
}

int Delay::getSourceStorageNumber(double time) {
	return currentStorageNumber = (currentStorageNumber - 1) % maxStorageCount;
}

int Delay::getSourceStorageNumberDelay(double time, int delayNumber) {
	printf("\nget source storage number for delay don't work!!!\n");
	return 0;
}

int Delay::getSourceStorageNumberDelayForDenseOutput(double time, int delayNumber) {
	printf("\nget source storage number for delay & for Dense output don't work!!!\n");
	return 0;
}

int Delay::getResultStorageNumber() {
	return currentStorageNumber;
}

double** Delay::getSource(int stage) {
	/*int sourceStorageNumber = getSourceStorageNumber(time);
	mSourceStorage[0] = mStepStorage[sourceStorageNumber]->getStageSource(stage);

	for (int i = 0; i < delayCount; ++i) {
		int sourceStorageNumberDelay = getSourceStorageNumberDelay(time, i);
		int sourceStorageNumberDelayForDenseOutput = getSourceStorageNumberDelayForDenseOutput(time, i);
		mStepStorage[sourceStorageNumberDelay]->getDenseOutput( mStepStorage[sourceStorageNumberDelayForDenseOutput], mSourceStorage[i+1] );
	}

	return mSourceStorage;*/
	return NULL;
}

double* Delay::getResult(int stage) {
	int resultStorageNumber = getResultStorageNumber();
	return mStepStorage[resultStorageNumber]->getStageResult(stage);
}

void Delay::prepareArgument(ProcessingUnit* pu, int stage, double timestep) {
	mStepStorage[currentStorageNumber]->prepareArgument(pu, stage, timestep);
}

double* Delay::getCurrentStateStageData(int stage) {
	return mStepStorage[currentStorageNumber]->getStageSource(stage);
}

double Delay::getStepError(ProcessingUnit* pu, double timestep) {
	return mStepStorage[currentStorageNumber]->getStepError(pu, timestep);
}

void Delay::confirmStep(ProcessingUnit* pu, double timestep) {
	mStepStorage[currentStorageNumber]->confirmStep(pu, timestep);
	currentStorageNumber = (currentStorageNumber + 1) % maxStorageCount;
}

void Delay::rejectStep(ProcessingUnit* pu, double timestep) {
	mStepStorage[currentStorageNumber]->rejectStep(pu, timestep);
}

void Delay::loadData(ProcessingUnit* pu, double* data) {
	mStepStorage[currentStorageNumber]->loadState(pu, data);
}

void Delay::getCurrentState(ProcessingUnit* pu, double* result) {
	mStepStorage[currentStorageNumber]->copyState(pu, result);
}

double* Delay::getCurrentStatePointer() {
	return mStepStorage[currentStorageNumber]->getStatePointer();
}

void Delay::saveState(ProcessingUnit* pu, char* path) {
	//mStepStorage->saveState(pu, path);
}

void Delay::loadState(ProcessingUnit* pu, char* path) {
	//mStepStorage->loadState(pu, path);
}
