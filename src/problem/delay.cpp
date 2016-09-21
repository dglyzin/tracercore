/*
 * delay.cpp
 *
 *  Created on: 16 дек. 2015 г.
 *      Author: frolov
 */

#include "delay.h"

using namespace std;

Delay::Delay(ProcessingUnit* _pu, int solverType, int count, double aTol, double rTol, int _delayCount) : ProblemType(_pu) {
	//TODO fix!!! реализация функции для ProcessingUnit!!!
	maxStorageCount = 1;

	mStepStorage = new StepStorage*[maxStorageCount];

	for (int i = 0; i < maxStorageCount; ++i) {
		mStepStorage[i] = createStageStorage(solverType, count, aTol, rTol);
	}

	delayCount = _delayCount;

	mSourceStorage = new double*[delayCount + 1];

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
	//return currentStorageNumber = (currentStorageNumber - 1) % maxStorageCount;
	return (currentStorageNumber + 1) % maxStorageCount;
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
	int sourceStorageNumber = getSourceStorageNumber(0.0);
	mSourceStorage[0] = mStepStorage[sourceStorageNumber]->getStageSource(stage);

	/*for (int i = 0; i < delayCount; ++i) {
		int sourceStorageNumberDelay = getSourceStorageNumberDelay(current, i);
		int sourceStorageNumberDelayForDenseOutput = getSourceStorageNumberDelayForDenseOutput(time, i);
		mStepStorage[sourceStorageNumberDelay]->getDenseOutput(mStepStorage[sourceStorageNumberDelayForDenseOutput],
				mSourceStorage[i + 1]);
	}*/

	return mSourceStorage;
	return NULL;
}

double* Delay::getResult(int stage) {
	int resultStorageNumber = getResultStorageNumber();
	return mStepStorage[resultStorageNumber]->getStageResult(stage);
}

void Delay::prepareArgument(int stage, double timestep) {
	mStepStorage[currentStorageNumber]->prepareArgument(stage, timestep);
}

double* Delay::getCurrentStateStageData(int stage) {
	return mStepStorage[currentStorageNumber]->getStageSource(stage);
}

double Delay::getStepError(double timestep) {
	return mStepStorage[currentStorageNumber]->getStepError(timestep);
}

void Delay::confirmStep(double timestep) {
	mStepStorage[currentStorageNumber]->confirmStep(timestep);
	currentStorageNumber = (currentStorageNumber + 1) % maxStorageCount;
}

void Delay::rejectStep(double timestep) {
	mStepStorage[currentStorageNumber]->rejectStep(timestep);
}

void Delay::loadData(double* data) {
	//mStepStorage[currentStorageNumber]->loadState(pu, data);
}

void Delay::getCurrentState(double* result) {
	mStepStorage[currentStorageNumber]->copyState(result);
}

double* Delay::getCurrentStatePointer() {
	return mStepStorage[currentStorageNumber]->getStatePointer();
}

void Delay::saveStateForDraw(char* path) {
	//mStepStorage->saveState(pu, path);

	mStepStorage[currentStorageNumber]->saveState(path);
}

void Delay::saveStateForLoad(char* path) {
	//mStepStorage->saveState(pu, path);
}

void Delay::loadState(std::ifstream& in) {
	//mStepStorage->loadState(pu, path);
}

bool Delay::isNan() {
	return false;
}

void Delay::print(int zCount, int yCount, int xCount, int cellSize) {
	return;
}
