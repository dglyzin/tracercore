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

double** Ordinary::getSource(int stage, double time) {
	return mStepStorage->getStageSource(stage);
}

double* Ordinary::getResult(int stage, double time) {
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
