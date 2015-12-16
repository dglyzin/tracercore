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

	mSourceStorage = new double* [delayCount];
}

Delay::~Delay() {
	for (int i = 0; i < maxStorageCount; ++i) {
		delete mStepStorage[i];
	}

	delete mSourceStorage;

	delete mSourceStorage;
}

