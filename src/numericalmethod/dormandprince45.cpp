/*
 * dp45.cpp
 *
 *  Created on: 22 февр. 2017 г.
 *      Author: frolov
 */

#include "dormandprince45.h"

using namespace std;

DormandPrince45::DormandPrince45(double _aTol, double _rTol) :
		NumericalMethod(_aTol, _rTol) {
}

DormandPrince45::~DormandPrince45() {
}

int DormandPrince45::getStageCount() {
	return 6;
}

bool DormandPrince45::isFSAL() {
	return true;
}

bool DormandPrince45::isErrorPermissible(double error, int totalDomainElements) {
	double err = sqrt(error / totalDomainElements);
	if (err < 1)
		return true;
	else
		return false;
}

bool DormandPrince45::isVariableStep() {
	return true;
}

double DormandPrince45::computeNewStep(double timeStep, double error, int totalDomainElements) {
	double err = sqrt(error / totalDomainElements);
	return timeStep * min(facmax, max(facmin, fac * pow(1.0 / err, 1.0 / 5.0)));
}

int DormandPrince45::getMemorySizePerState(int elementCount) {
	// TODO: 2 заменить на что-то более вразумительное
	return (elementCount * SIZE_DOUBLE) * (1 + KSTORAGE_COUNT);
}

int DormandPrince45::getKStorageCount() {
	return KSTORAGE_COUNT;
}

int DormandPrince45::getCommonTempStorageCount() {
	return COMMON_TEMP_STROTAGE_COUNT;
}

double* DormandPrince45::getStorageResult(double* state, double** kStorages, double** commonTempStorages,
		int stageNumber) {
	switch (stageNumber) {
		case 0:
			return kStorages[K2];
		case 1:
			return kStorages[K3];
		case 2:
			return kStorages[K4];
		case 3:
			return kStorages[K5];
		case 4:
			return kStorages[K6];
		case 5:
			return kStorages[K7];
		case SOLVER_INIT_STAGE:
			return kStorages[K1];
		default:
			assert(0);
			return NULL;
	}
}

double* DormandPrince45::getStorageSource(double* state, double** kStorages, double** commonTempStorages,
		int stageNumber) {
	switch (stageNumber) {
		case 0:
		case 1:
		case 2:
		case 3:
		case 4:
		case 5:
			return commonTempStorages[ARG];
		case SOLVER_INIT_STAGE:
			return state;
		default:
			assert(0);
			return NULL;
	}
}

double DormandPrince45::getStageTimeStepCoefficient(int stageNumber) {
	switch (stageNumber) {
		case 0:
			return c2;
		case 1:
			return c3;
		case 2:
			return c4;
		case 3:
			return c5;
		case 4:
			return 1.0;
		case 5:
			return 1.0;
		case SOLVER_INIT_STAGE:
			return 0.0;
		default:
			assert(0);
			return 0.0;
	}
}

void DormandPrince45::prepareArgument(ProcessingUnit* pu, double* state, double** kStorages,
		double** commonTempStorages, double timeStep, int stageNumber, int size) {
	switch (stageNumber) {
		case SOLVER_INIT_STAGE:
			pu->multiplyArrayByNumberAndSum(commonTempStorages[ARG], kStorages[K1], timeStep * a21, state, size);
			//prepareFSAL(timestep);
			break;
		case 0:
			break;
		case 1:
			pu->multiplyArrayByNumber(commonTempStorages[ARG], kStorages[K1], a31, size);
			pu->multiplyArrayByNumberAndSum(commonTempStorages[ARG], kStorages[K2], a32, commonTempStorages[ARG], size);
			pu->multiplyArrayByNumber(commonTempStorages[ARG], commonTempStorages[ARG], timeStep, size);
			pu->sumArrays(commonTempStorages[ARG], commonTempStorages[ARG], state, size);
			break;
		case 2:
			pu->multiplyArrayByNumber(commonTempStorages[ARG], kStorages[K1], a41, size);
			pu->multiplyArrayByNumberAndSum(commonTempStorages[ARG], kStorages[K2], a42, commonTempStorages[ARG], size);
			pu->multiplyArrayByNumberAndSum(commonTempStorages[ARG], kStorages[K3], a43, commonTempStorages[ARG], size);
			pu->multiplyArrayByNumber(commonTempStorages[ARG], commonTempStorages[ARG], timeStep, size);
			pu->sumArrays(commonTempStorages[ARG], commonTempStorages[ARG], state, size);
			break;
		case 3:
			pu->multiplyArrayByNumber(commonTempStorages[ARG], kStorages[K1], a51, size);
			pu->multiplyArrayByNumberAndSum(commonTempStorages[ARG], kStorages[K2], a52, commonTempStorages[ARG], size);
			pu->multiplyArrayByNumberAndSum(commonTempStorages[ARG], kStorages[K3], a53, commonTempStorages[ARG], size);
			pu->multiplyArrayByNumberAndSum(commonTempStorages[ARG], kStorages[K4], a54, commonTempStorages[ARG], size);
			pu->multiplyArrayByNumber(commonTempStorages[ARG], commonTempStorages[ARG], timeStep, size);
			pu->sumArrays(commonTempStorages[ARG], commonTempStorages[ARG], state, size);
			break;
		case 4:
			pu->multiplyArrayByNumber(commonTempStorages[ARG], kStorages[K1], a61, size);
			pu->multiplyArrayByNumberAndSum(commonTempStorages[ARG], kStorages[K2], a62, commonTempStorages[ARG], size);
			pu->multiplyArrayByNumberAndSum(commonTempStorages[ARG], kStorages[K3], a63, commonTempStorages[ARG], size);
			pu->multiplyArrayByNumberAndSum(commonTempStorages[ARG], kStorages[K4], a64, commonTempStorages[ARG], size);
			pu->multiplyArrayByNumberAndSum(commonTempStorages[ARG], kStorages[K5], a65, commonTempStorages[ARG], size);
			pu->multiplyArrayByNumber(commonTempStorages[ARG], commonTempStorages[ARG], timeStep, size);
			pu->sumArrays(commonTempStorages[ARG], commonTempStorages[ARG], state, size);
			break;
		case 5:
			pu->multiplyArrayByNumber(commonTempStorages[ARG], kStorages[K1], a71, size);
			pu->multiplyArrayByNumberAndSum(commonTempStorages[ARG], kStorages[K2], a73, commonTempStorages[ARG], size);
			pu->multiplyArrayByNumberAndSum(commonTempStorages[ARG], kStorages[K4], a74, commonTempStorages[ARG], size);
			pu->multiplyArrayByNumberAndSum(commonTempStorages[ARG], kStorages[K5], a75, commonTempStorages[ARG], size);
			pu->multiplyArrayByNumberAndSum(commonTempStorages[ARG], kStorages[K6], a76, commonTempStorages[ARG], size);
			pu->multiplyArrayByNumber(commonTempStorages[ARG], commonTempStorages[ARG], timeStep, size);
			pu->sumArrays(commonTempStorages[ARG], commonTempStorages[ARG], state, size);
			break;
		default:
			assert(0);
			break;
	}
}

void DormandPrince45::confirmStep(ProcessingUnit* pu, ISmartCopy* sc, double** sourceState, double** sourceKStorages,
		double** destinationState, double** destinationKStorages, double** commonTempStorages, double timeStep,
		int size) {
	/*double* temp = mState;
	 mState = mArg;
	 mArg = temp;

	 temp = mTempStore7;
	 mTempStore7 = mTempStore1;
	 mTempStore1 = temp;*/

	pu->swapStorages(sourceState, commonTempStorages + ARG);
	sc->swapCopy(pu, sourceKStorages + K7, destinationKStorages + K1, size);
}

void DormandPrince45::rejectStep(ProcessingUnit* pu, double* state, double** kStorages, double** commonTempStorages,
		double timeStep, int size) {
}

double DormandPrince45::computeStepError(ProcessingUnit* pu, double** kStorages, double** commonTempStorages,
		double timeStep, int size) {
}

void DormandPrince45::computeDenseOutput(ProcessingUnit* pu, double* state, double** kStorages, double timeStep,
		double theta, double* result, int size) {
}
