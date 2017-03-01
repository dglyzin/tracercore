/*
 * ordinary.cpp
 *
 *  Created on: 24 сент. 2015 г.
 *      Author: frolov
 */

#include "../problem_old/ordinary.h"

using namespace std;

Ordinary::Ordinary(ProcessingUnit* _pu, int solverType, int count, double aTol, double rTol) : ProblemType(_pu) {
	mStepStorage = createStageStorage(solverType, count, aTol, rTol);

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

void Ordinary::prepareArgument(int stage, double timestep) {
	mStepStorage->prepareArgument(stage, timestep);
}

double* Ordinary::getCurrentStateStageData(int stage) {
	return mStepStorage->getStageSource(stage);
}

double Ordinary::getStepError(double timestep) {
	return mStepStorage->getStepError(timestep);
}

void Ordinary::confirmStep(double timestep) {
	mStepStorage->confirmStep(timestep);
}

void Ordinary::rejectStep(double timestep) {
	mStepStorage->rejectStep(timestep);
}

void Ordinary::loadData(double* data) {
	//mStepStorage->loadState(pu, data);
	printf("\nLOAD DATA ORDINARY DEPRICATED!!!\n");
}

void Ordinary::getCurrentState(double* result) {
	mStepStorage->copyState(result);
}

double* Ordinary::getCurrentStatePointer() {
	return mStepStorage->getStatePointer();
}

void Ordinary::saveStateForDraw(char* path) {
	mStepStorage->saveState(path);
}

void Ordinary::saveStateForLoad(char* path) {
	mStepStorage->saveState(path);
}

void Ordinary::saveStateForDrawDenseOutput(char* path, double timestep, double tetha) {
	mStepStorage->saveDenseOutput(path, timestep, tetha);
}

void Ordinary::loadState(ifstream& in) {
	mStepStorage->loadState(in);
}

bool Ordinary::isNan() {
	return mStepStorage->isNan();
}

void Ordinary::print(int zCount, int yCount, int xCount, int cellSize) {
	mStepStorage->print(zCount, yCount, xCount, cellSize);
}
