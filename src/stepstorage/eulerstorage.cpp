/*
 * eulerstorage.cpp
 *
 *  Created on: 24 сент. 2015 г.
 *      Author: frolov
 */

#include "eulerstorage.h"

using namespace std;

EulerStorage::EulerStorage() :
		StepStorage() {
	mTempStore1 = NULL;
}

EulerStorage::EulerStorage(ProcessingUnit* pu, int count, double _aTol, double _rTol) :
		StepStorage(pu, count, _aTol, _rTol) {
	mTempStore1 = pu->newDoubleArray(mCount);
}

EulerStorage::~EulerStorage() {
}

void EulerStorage::saveMTempStores(ProcessingUnit* pu, char* path) {
	pu->saveArray(mTempStore1, mCount, path);
}

void EulerStorage::loadMTempStores(ProcessingUnit* pu, ifstream& in) {
	pu->loadArray(mTempStore1, mCount, in);
}

int EulerStorage::getSizeChild(int elementCount) {
	int size = 0;

	size += elementCount * SIZE_DOUBLE; // mTempStore1

	return size;
}

double* EulerStorage::getStageSource(int stahe) {
	return mState;
}

double* EulerStorage::getStageResult(int stage) {
	return mTempStore1;
}

double EulerStorage::getStageTimeStep(int stage) {
	return 0.0;
}

void EulerStorage::prepareArgument(ProcessingUnit* pu, int stage, double timestep) {
	pu->multiplyArrayByNumber(mTempStore1, mTempStore1, timestep, mCount);
	pu->sumArrays(mTempStore1, mState, mTempStore1, mCount);
}

void EulerStorage::confirmStep(ProcessingUnit* pu, double timestep) {
	double* temp = mState;
	mState = mTempStore1;
	mTempStore1 = temp;
}

void EulerStorage::rejectStep(ProcessingUnit* pu, double timestep) {
	return;
}

double EulerStorage::getStepError(ProcessingUnit* pu, double timestep) {
	return 0.0;
}

bool EulerStorage::isFSAL() {
	return false;
}

bool EulerStorage::isVariableStep() {
	return false;
}

int EulerStorage::getStageCount() {
	return 1;
}

double EulerStorage::getNewStep(double timestep, double error, int totalDomainElements) {
	return timestep;
}

bool EulerStorage::isErrorPermissible(double error, int totalDomainElements) {
	return true;
}

void EulerStorage::getDenseOutput(ProcessingUnit* pu, double timestep, double tetha, double* result) {
	printf("\nEuler dense output DON'T WORK!\n");
}

void EulerStorage::print(ProcessingUnit* pu, int zCount, int yCount, int xCount, int cellSize) {
	printf("mState:\n");
	pu->printArray(mState, zCount, yCount, xCount, cellSize);

	printf("\nmTempStore1:\n");
	pu->printArray(mTempStore1, zCount, yCount, xCount, cellSize);
}
