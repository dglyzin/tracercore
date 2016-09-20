/*
 * ordinary.cpp
 *
 *  Created on: 24 сент. 2015 г.
 *      Author: frolov
 */

#include "ordinary.h"

using namespace std;

Ordinary::Ordinary(ProcessingUnit* pu, int solverType, int count, double aTol, double rTol) {
	mStepStorage = createStageStorage(pu, solverType, count, aTol, rTol);

	mSourceStorage = new double*[1];
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
	mStepStorage->prepareArgument(stage, timestep);
}

double* Ordinary::getCurrentStateStageData(int stage) {
	return mStepStorage->getStageSource(stage);
}

double Ordinary::getStepError(ProcessingUnit* pu, double timestep) {
	return mStepStorage->getStepError(timestep);
}

void Ordinary::confirmStep(ProcessingUnit* pu, double timestep) {
	mStepStorage->confirmStep(timestep);
}

void Ordinary::rejectStep(ProcessingUnit* pu, double timestep) {
	mStepStorage->rejectStep(timestep);
}

void Ordinary::loadData(ProcessingUnit* pu, double* data) {
	//mStepStorage->loadState(pu, data);
	printf("\nLOAD DATA ORDINARY DEPRICATED!!!\n");
}

void Ordinary::getCurrentState(ProcessingUnit* pu, double* result) {
	mStepStorage->copyState(result);
}

double* Ordinary::getCurrentStatePointer() {
	return mStepStorage->getStatePointer();
}

void Ordinary::saveStateForDraw(ProcessingUnit* pu, char* path) {
	mStepStorage->saveState(path);
}

void Ordinary::saveStateForLoad(ProcessingUnit* pu, char* path) {
	mStepStorage->saveState(path);
}

void Ordinary::loadState(ProcessingUnit* pu, ifstream& in) {
	mStepStorage->loadState(in);
}

bool Ordinary::isNan(ProcessingUnit* pu) {
	return mStepStorage->isNan();
}

void Ordinary::print(ProcessingUnit* pu, int zCount, int yCount, int xCount, int cellSize) {
	mStepStorage->print(zCount, yCount, xCount, cellSize);
}
