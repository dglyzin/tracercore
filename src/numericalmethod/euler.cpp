/*
 * euler.cpp
 *
 *  Created on: 13 февр. 2017 г.
 *      Author: frolov
 */

#include "euler.h"

Euler::Euler(double _aTol, double _rTol) :
		NumericalMethod(_aTol, _rTol) {
}

Euler::~Euler() {
}

int Euler::getStageCount() {
	return 1;
}

bool Euler::isFSAL() {
	return false;
}

bool Euler::isErrorPermissible(double error, int totalDomainElements) {
	return true;
}

bool Euler::isVariableStep() {
	return false;
}

double Euler::computeNewStep(double timeStep, double error, int totalDomainElements) {
	return timeStep;
}

int Euler::getKStorageCount() {
	return KSTORAGE_COUNT;
}

int Euler::getCommonTempStorageCount() {
	return 0;
}

double* Euler::getStorageResult(double* state, double** kStorages, double** commonTempStorages, int stageNumber) {
	return kStorages[K1];
}

double* Euler::getStorageSource(double* state, double** kStorages, double** commonTempStorages, int stageNumber) {
	return state;
}

double Euler::getStageTimeStepCoefficient(int stageNumber) {
	// TODO: точно?
	return 1.0;
}

void Euler::prepareArgument(ProcessingUnit* pu, double* state, double** kStorages, double** commonTempStorages,
		double timeStep, int stageNumber, int size) {
	return;
}

void Euler::confirmStep(ProcessingUnit* pu, ISmartCopy* sc, double** sourceState, double** sourceKStorages,
		double** destinationState, double** destinationKStorages, double** commonTempStorages, double timeStep,
		int size) {
	/*pu->multiplyArrayByNumber(mTempStore1, mTempStore1, timestep, mCount);
	 pu->sumArrays(mTempStore1, mState, mTempStore1, mCount);

	 double* temp = mState;
	 mState = mTempStore1;
	 mTempStore1 = temp;*/
	pu->multiplyArrayByNumberAndSum(*destinationState, sourceKStorages[K1], timeStep, *sourceState, size);
}

void Euler::rejectStep(ProcessingUnit* pu, double* state, double** kStorages, double** commonTempStorages,
		double timeStep, int size) {
	return;
}

double Euler::computeStepError(ProcessingUnit* pu, double* state, double** kStorages, double** commonTempStorages,
		double timeStep, int size) {
	return 0.0;
}

void Euler::computeDenseOutput(ProcessingUnit* pu, double* state, double** kStorages, double** commonTempStorages,
		double timeStep, double theta, double* result, int size) {
	pu->multiplyArrayByNumberAndSum(result, kStorages[K1], theta * timeStep, state, size);
}
