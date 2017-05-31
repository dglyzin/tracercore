/*
 * rungekutta.cpp
 *
 *  Created on: 14 мар. 2017 г.
 *      Author: frolov
 */

#include "rungekutta4.h"

RungeKutta4::RungeKutta4(double _aTol, double _rTol) :
		NumericalMethod(_aTol, _rTol) {
}

RungeKutta4::~RungeKutta4() {
}

int RungeKutta4::getStageCount() {
	return 4;
}

bool RungeKutta4::isFSAL() {
	return false;
}

bool RungeKutta4::isErrorPermissible(double error, int totalDomainElements) {
	return true;
}

bool RungeKutta4::isVariableStep() {
	return false;
}

double RungeKutta4::computeNewStep(double timeStep, double error, int totalDomainElements) {
	return timeStep;
}

int RungeKutta4::getKStorageCount() {
	return KSTORAGE_COUNT;
}

int RungeKutta4::getCommonTempStorageCount() {
	return COMMON_TEMP_STROTAGE_COUNT;
}

double* RungeKutta4::getStorageResult(double* state, double** kStorages, double** commonTempStorages, int stageNumber) {
	switch (stageNumber) {
		case 0:
			return kStorages[K1];
		case 1:
			return kStorages[K2];
		case 2:
			return kStorages[K3];
		case 3:
			return kStorages[K4];
		default:
			assert(0);
			return NULL;
	}
}

double* RungeKutta4::getStorageSource(double* state, double** kStorages, double** commonTempStorages, int stageNumber) {
	switch (stageNumber) {
		case 0:
			return state;
		case 1:
		case 2:
		case 3:
			return commonTempStorages[ARG];
		default:
			assert(0);
			return NULL;
	}
}

double RungeKutta4::getStageTimeStepCoefficient(int stageNumber) {
	switch (stageNumber) {
		case 0:
			return 0.0;
		case 1:
			return 0.5;
		case 2:
			return 0.5;
		case 3:
			return 1.0;
		default:
			assert(0);
			return 0.0;
	}
}

void RungeKutta4::prepareArgument(ProcessingUnit* pu, double* state, double** kStorages, double** commonTempStorages,
		double timeStep, int stageNumber, int size) {
	switch (stageNumber) {
		case 0:
			break;
		case 1: //0
			pu->multiplyArrayByNumber(commonTempStorages[ARG], kStorages[K1], 0.5 * timeStep, size);
			pu->sumArrays(commonTempStorages[ARG], commonTempStorages[ARG], state, size);
			break;
		case 2:
			pu->multiplyArrayByNumber(commonTempStorages[ARG], kStorages[K2], 0.5 * timeStep, size);
			pu->sumArrays(commonTempStorages[ARG], commonTempStorages[ARG], state, size);
			break;
		case 3:
			pu->multiplyArrayByNumber(commonTempStorages[ARG], kStorages[K3], 1.0 * timeStep, size);
			pu->sumArrays(commonTempStorages[ARG], commonTempStorages[ARG], state, size);
			break;
			//case 0:
			/*pc->multiplyArrayByNumber(mTempStore1, mTempStore1, b1, mCount);
			 pc->multiplyArrayByNumber(mTempStore2, mTempStore2, b2, mCount);
			 pc->multiplyArrayByNumber(mTempStore3, mTempStore3, b3, mCount);
			 pc->multiplyArrayByNumber(mTempStore4, mTempStore4, b4, mCount);

			 pc->sumArrays(mArg, mTempStore1, mTempStore2, mCount);
			 pc->sumArrays(mArg, mArg, mTempStore3, mCount);
			 pc->sumArrays(mArg, mArg, mTempStore4, mCount);

			 pc->multiplyArrayByNumber(mArg, mArg, timestep, mCount);

			 pc->sumArrays(mArg, mArg, mState, mCount);*/

			/*pu->multiplyArrayByNumber(mArg, mTempStore1, b1, mCount);
			 pu->multiplyArrayByNumberAndSum(mArg, mTempStore2, b2, mArg, mCount);
			 pu->multiplyArrayByNumberAndSum(mArg, mTempStore3, b3, mArg, mCount);
			 pu->multiplyArrayByNumberAndSum(mArg, mTempStore4, b4, mArg, mCount);

			 pu->multiplyArrayByNumber(mArg, mArg, timestep, mCount);

			 pu->sumArrays(mArg, state, mArg, mCount);*/

			//break;
		default:
			break;
	}
}

void RungeKutta4::confirmStep(ProcessingUnit* pu, ISmartCopy* sc, double** sourceState, double** sourceKStorages,
		double** destinationState, double** destinationKStorages, double** commonTempStorages, double timeStep,
		int size) {
	pu->multiplyArrayByNumber(commonTempStorages[ARG], sourceKStorages[K1], b1, size);
	pu->multiplyArrayByNumberAndSum(commonTempStorages[ARG], sourceKStorages[K2], b2, commonTempStorages[ARG], size);
	pu->multiplyArrayByNumberAndSum(commonTempStorages[ARG], sourceKStorages[K3], b3, commonTempStorages[ARG], size);
	pu->multiplyArrayByNumberAndSum(commonTempStorages[ARG], sourceKStorages[K4], b4, commonTempStorages[ARG], size);

	pu->multiplyArrayByNumber(commonTempStorages[ARG], commonTempStorages[ARG], timeStep, size);

	pu->sumArrays(commonTempStorages[ARG], *sourceState, commonTempStorages[ARG], size);

	pu->swapArray(destinationState, commonTempStorages + ARG);
}

void RungeKutta4::rejectStep(ProcessingUnit* pu, double* state, double** kStorages, double** commonTempStorages,
		double timeStep, int size) {
	return;
}

double RungeKutta4::computeStepError(ProcessingUnit* pu, double* state, double** kStorages, double** commonTempStorages,
		double timeStep, int size) {
	return 0.0;
}

void RungeKutta4::computeDenseOutput(ProcessingUnit* pu, double* state, double** kStorages, double** commonTempStorages,
		double timeStep, double theta, double* result, int size) {
	pu->copyArray(state, result, size);
}
