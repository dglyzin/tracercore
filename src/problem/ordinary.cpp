/*
 * ordinary.cpp
 *
 *  Created on: 24 сент. 2015 г.
 *      Author: frolov
 */

#include "ordinary.h"

Ordinary::Ordinary(ProcessingUnit* pu, int solverType, int count, double aTol, double rTol) {
	mStepStorage = createStageStorage(pu, solverType, count, aTol, rTol);
}

Ordinary::~Ordinary() {
	delete mStepStorage;
}

double* Ordinary::getSource(int stage, double time) {
	return mStepStorage->getStageSource(stage);
}

double* Ordinary::getResult(int stage, double time) {
	return mStepStorage->getStageResult(stage);
}

void Ordinary::prepareArgument(ProcessingUnit* pu, int stage, double timestep) {
	mStepStorage->prepareArgument(pu, stage, timestep);
}
