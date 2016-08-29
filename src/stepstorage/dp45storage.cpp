/*
 * dp45storage.cpp
 *
 *  Created on: 06 окт. 2015 г.
 *      Author: frolov
 */

#include "dp45storage.h"

using namespace std;

DP45Storage::DP45Storage() :
		StepStorage() {
	mTempStore1 = NULL;
	mTempStore2 = NULL;
	mTempStore3 = NULL;
	mTempStore4 = NULL;
	mTempStore5 = NULL;
	mTempStore6 = NULL;
	mTempStore7 = NULL;

	mArg = NULL;

	temp = NULL;
	temp2 = NULL;
}

DP45Storage::DP45Storage(ProcessingUnit* pu, int count, double _aTol, double _rTol) :
		StepStorage(pu, count, _aTol, _rTol) {
	mTempStore1 = pu->newDoubleArray(mCount);
	mTempStore2 = pu->newDoubleArray(mCount);
	mTempStore3 = pu->newDoubleArray(mCount);
	mTempStore4 = pu->newDoubleArray(mCount);
	mTempStore5 = pu->newDoubleArray(mCount);
	mTempStore6 = pu->newDoubleArray(mCount);
	mTempStore7 = pu->newDoubleArray(mCount);

	mArg = pu->newDoubleArray(mCount);

	temp = pu->newDoubleArray(mCount);
	temp2 = pu->newDoubleArray(mCount);
}

DP45Storage::~DP45Storage() {
}

void DP45Storage::prepareFSAL(ProcessingUnit* pu, double timestep) {
	/*#pragma omp parallel for
	 for (int idx = 0; idx < mCount; idx++)
	 mArg[idx] = mState[idx] + a21 * timeStep * mTempStore1[idx];*/
	pu->multiplyArrayByNumberAndSum(mArg, mTempStore1, a21 * timestep, mState, mCount);
}

void DP45Storage::saveMTempStores(ProcessingUnit* pu, char* path) {
	pu->saveArray(mTempStore1, mCount, path);
	pu->saveArray(mTempStore2, mCount, path);
	pu->saveArray(mTempStore3, mCount, path);
	pu->saveArray(mTempStore4, mCount, path);
	pu->saveArray(mTempStore5, mCount, path);
	pu->saveArray(mTempStore6, mCount, path);
	pu->saveArray(mTempStore7, mCount, path);

	pu->saveArray(mArg, mCount, path);
}

void DP45Storage::loadMTempStores(ProcessingUnit* pu, ifstream& in) {
	pu->loadArray(mTempStore1, mCount, in);
	pu->loadArray(mTempStore2, mCount, in);
	pu->loadArray(mTempStore3, mCount, in);
	pu->loadArray(mTempStore4, mCount, in);
	pu->loadArray(mTempStore5, mCount, in);
	pu->loadArray(mTempStore6, mCount, in);
	pu->loadArray(mTempStore7, mCount, in);

	pu->loadArray(mArg, mCount, in);
}

int DP45Storage::getSizeChild(int elementCount) {
	int size = 0;

	size += elementCount * SIZE_DOUBLE; // mTempStore1
	size += elementCount * SIZE_DOUBLE; // mTempStore2
	size += elementCount * SIZE_DOUBLE; // mTempStore3
	size += elementCount * SIZE_DOUBLE; // mTempStore4
	size += elementCount * SIZE_DOUBLE; // mTempStore5
	size += elementCount * SIZE_DOUBLE; // mTempStore6
	size += elementCount * SIZE_DOUBLE; // mTempStore7
	size += elementCount * SIZE_DOUBLE; // mArg

	size += elementCount * SIZE_DOUBLE; // temp
	size += elementCount * SIZE_DOUBLE; // temp2

	return size;
}

double DP45Storage::getB1(double theta) {
	return pow(theta, 2) * (3 - 2 * theta) * a71 + theta * pow(theta - 1, 2)
			- pow(theta, 2) * pow(theta - 1, 2) * 5 * (2558722523 - 31403016 * theta) / 11282082432;
}

double DP45Storage::getB3(double theta) {
	return pow(theta, 2) * (3 - 2 * theta) * a73 + pow(theta, 2) * pow(theta - 1, 2) * 100 * (882725551 - 15701508 * theta) / 32700410799;
}

double DP45Storage::getB4(double theta) {
	return pow(theta, 2) * (3 - 2 * theta) * a74
			- pow(theta, 2) * pow(theta - 1, 2) * 25 * (443332067 - 31403016 * theta) / 1880347072;
}

double DP45Storage::getB5(double theta) {
	return pow(theta, 2) * (3 - 2 * theta) * a75
			+ pow(theta, 2) * pow(theta - 1, 2) * 32805 * (23143187 - 3489224 * theta) / 199316789632;
}

double DP45Storage::getB6(double theta) {
	return pow(theta, 2) * (3 - 2 * theta) * a76
			- pow(theta, 2) * pow(theta - 1, 2) * 55 * (29972135 - 7076736 * theta) / 822651844;
}

double DP45Storage::getB7(double theta) {
	/*return pow(theta, 2) * (theta - 1) * b7
			+ pow(theta, 2) * pow(theta - 1, 2) * 10 * (7414447 - 829305 * theta) / 29380423;*/
}

double* DP45Storage::getStageSource(int stage) {
	/*if      (stage == 0) return mArg;
	 else if (stage == 1) return mArg;
	 else if (stage == 2) return mArg;
	 else if (stage == 3) return mArg;
	 else if (stage == 4) return mArg;
	 else if (stage == 5) return mArg;
	 else if (stage == -1) return mState;
	 else assert(0);
	 return NULL;*/

	switch (stage) {
		case 0:
		case 1:
		case 2:
		case 3:
		case 4:
		case 5:
			return mArg;
		case SOLVER_INIT_STAGE:
			return mState;
		default:
			assert(0);
			return NULL;
	}
}

double* DP45Storage::getStageResult(int stage) {
	/*if      (stage == 0) return mTempStore2;
	 else if (stage == 1) return mTempStore3;
	 else if (stage == 2) return mTempStore4;
	 else if (stage == 3) return mTempStore5;
	 else if (stage == 4) return mTempStore6;
	 else if (stage == 5) return mTempStore7;
	 else if (stage == -1) return mTempStore1;
	 else assert(0);
	 return NULL;*/

	switch (stage) {
		case 0:
			return mTempStore2;
		case 1:
			return mTempStore3;
		case 2:
			return mTempStore4;
		case 3:
			return mTempStore5;
		case 4:
			return mTempStore6;
		case 5:
			return mTempStore7;
		case SOLVER_INIT_STAGE:
			return mTempStore1;
		default:
			assert(0);
			return NULL;
	}
}

double DP45Storage::getStageTimeStep(int stage) {
	/*if      (stage == 0) return c2;
	 else if (stage == 1) return c3;
	 else if (stage == 2) return c4;
	 else if (stage == 3) return c5;
	 else if (stage == 4) return 1.0;
	 else if (stage == 5) return 1.0;
	 else if (stage ==-1) return 0.0;
	 else assert(0);
	 return 0.0;*/

	switch (stage) {
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

void DP45Storage::prepareArgument(ProcessingUnit* pu, int stage, double timestep) {
	/*if      (stage == 0)
	 #pragma omp parallel for
	 for (int idx=0; idx<mCount; idx++)
	 mArg[idx] = mState[idx]+timeStep*(a31*mTempStore1[idx] + a32*mTempStore2[idx]);
	 else if (stage == 1)
	 #pragma omp parallel for
	 for (int idx=0; idx<mCount; idx++)
	 mArg[idx] = mState[idx]+timeStep*(a41*mTempStore1[idx] + a42*mTempStore2[idx] + a43*mTempStore3[idx]);
	 else if (stage == 2)
	 #pragma omp parallel for
	 for (int idx=0; idx<mCount; idx++)
	 mArg[idx] = mState[idx]+timeStep*(a51*mTempStore1[idx] + a52*mTempStore2[idx] + a53*mTempStore3[idx] + a54*mTempStore4[idx]);
	 else if (stage == 3)
	 #pragma omp parallel for
	 for (int idx=0; idx<mCount; idx++)
	 mArg[idx] = mState[idx]+timeStep*(a61*mTempStore1[idx] + a62*mTempStore2[idx] + a63*mTempStore3[idx] + a64*mTempStore4[idx] +a65*mTempStore5[idx]);
	 else if (stage == 4)
	 #pragma omp parallel for
	 for (int idx=0; idx<mCount; idx++)
	 mArg[idx] = mState[idx]+timeStep*(a71*mTempStore1[idx] + a73*mTempStore3[idx] + a74*mTempStore4[idx] + a75*mTempStore5[idx] +a76*mTempStore6[idx]);
	 else if (stage == 5)
	 { //nothing to be done here before step is confirmed, moved to confirmStep
	 }
	 else if (stage == -1)
	 #pragma omp parallel for
	 for (int idx=0; idx<mCount; idx++)
	 mArg[idx] = mState[idx]+a21*timeStep*mTempStore1[idx];

	 else assert(0);*/

	switch (stage) {
		case 0:
			pu->multiplyArrayByNumber(mArg, mTempStore1, a31, mCount);
			pu->multiplyArrayByNumberAndSum(mArg, mTempStore2, a32, mArg, mCount);
			pu->multiplyArrayByNumber(mArg, mArg, timestep, mCount);
			pu->sumArrays(mArg, mArg, mState, mCount);
			break;
		case 1:
			pu->multiplyArrayByNumber(mArg, mTempStore1, a41, mCount);
			pu->multiplyArrayByNumberAndSum(mArg, mTempStore2, a42, mArg, mCount);
			pu->multiplyArrayByNumberAndSum(mArg, mTempStore3, a43, mArg, mCount);
			pu->multiplyArrayByNumber(mArg, mArg, timestep, mCount);
			pu->sumArrays(mArg, mArg, mState, mCount);
			break;
		case 2:
			pu->multiplyArrayByNumber(mArg, mTempStore1, a51, mCount);
			pu->multiplyArrayByNumberAndSum(mArg, mTempStore2, a52, mArg, mCount);
			pu->multiplyArrayByNumberAndSum(mArg, mTempStore3, a53, mArg, mCount);
			pu->multiplyArrayByNumberAndSum(mArg, mTempStore4, a54, mArg, mCount);
			pu->multiplyArrayByNumber(mArg, mArg, timestep, mCount);
			pu->sumArrays(mArg, mArg, mState, mCount);
			break;
		case 3:
			pu->multiplyArrayByNumber(mArg, mTempStore1, a61, mCount);
			pu->multiplyArrayByNumberAndSum(mArg, mTempStore2, a62, mArg, mCount);
			pu->multiplyArrayByNumberAndSum(mArg, mTempStore3, a63, mArg, mCount);
			pu->multiplyArrayByNumberAndSum(mArg, mTempStore4, a64, mArg, mCount);
			pu->multiplyArrayByNumberAndSum(mArg, mTempStore5, a65, mArg, mCount);
			pu->multiplyArrayByNumber(mArg, mArg, timestep, mCount);
			pu->sumArrays(mArg, mArg, mState, mCount);
			break;
		case 4:
			pu->multiplyArrayByNumber(mArg, mTempStore1, a71, mCount);
			pu->multiplyArrayByNumberAndSum(mArg, mTempStore3, a73, mArg, mCount);
			pu->multiplyArrayByNumberAndSum(mArg, mTempStore4, a74, mArg, mCount);
			pu->multiplyArrayByNumberAndSum(mArg, mTempStore5, a75, mArg, mCount);
			pu->multiplyArrayByNumberAndSum(mArg, mTempStore6, a76, mArg, mCount);
			pu->multiplyArrayByNumber(mArg, mArg, timestep, mCount);
			pu->sumArrays(mArg, mArg, mState, mCount);
			break;
		case 5:
			break;
		case SOLVER_INIT_STAGE:
			///pu->multiplyArrayByNumberAndSum(mArg, mTempStore1, timestep * a21, mState, mCount);
			prepareFSAL(pu, timestep);
			break;
		default:
			assert(0);
			break;
	}
}

void DP45Storage::confirmStep(ProcessingUnit* pu, double timestep) {
	double* temp = mState;
	mState = mArg;
	mArg = temp;

	temp = mTempStore7;
	mTempStore7 = mTempStore1;
	mTempStore1 = temp;

	prepareFSAL(pu, timestep);
}

void DP45Storage::rejectStep(ProcessingUnit* pu, double timestep) {
	prepareFSAL(pu, timestep);
}

double DP45Storage::getStepError(ProcessingUnit* pu, double timestep) {
	/*double err=0;
	 #pragma omp parallel for reduction (+:err)
	 for (int idx=0; idx<mCount; idx++){
	 double erri =  timeStep * (e1 * mTempStore1[idx] + e3 * mTempStore3[idx] + e4 * mTempStore4[idx] +
	 e5 * mTempStore5[idx] + e6 * mTempStore6[idx]+ e7 * mTempStore7[idx])
	 /(aTol + rTol * max(mArg[idx], mState[idx]));
	 err += erri * erri;
	 }

	 return err;*/

	/*double* temp = mTempStore2;
	 double* temp2 = mTempStore3;*/

	pu->multiplyArrayByNumber(temp, mTempStore1, timestep * e1, mCount);
	pu->multiplyArrayByNumberAndSum(temp, mTempStore3, timestep * e3, temp, mCount);
	pu->multiplyArrayByNumberAndSum(temp, mTempStore4, timestep * e4, temp, mCount);
	pu->multiplyArrayByNumberAndSum(temp, mTempStore5, timestep * e5, temp, mCount);
	pu->multiplyArrayByNumberAndSum(temp, mTempStore6, timestep * e6, temp, mCount);
	pu->multiplyArrayByNumberAndSum(temp, mTempStore7, timestep * e7, temp, mCount);

//pu->multiplyArrayByNumber(temp, temp, timestep, mCount);

	pu->maxElementsElementwise(temp2, mArg, mState, mCount);
	pu->multiplyArrayByNumber(temp2, temp2, rTol, mCount);
	pu->addNumberToArray(temp2, temp2, aTol, mCount);

	pu->divisionArraysElementwise(temp, temp, temp2, mCount);

	pu->multiplyArraysElementwise(temp, temp, temp, mCount);

	return pu->sumArrayElements(temp, mCount);
}

bool DP45Storage::isFSAL() {
	return true;
}

bool DP45Storage::isVariableStep() {
	return true;
}

int DP45Storage::getStageCount() {
	return 6;
}

double DP45Storage::getNewStep(double timestep, double error, int totalDomainElements) {
	double err = sqrt(error / totalDomainElements);
	return timestep * min(facmax, max(facmin, fac * pow(1.0 / err, 1.0 / 5.0)));
}

bool DP45Storage::isErrorPermissible(double error, int totalDomainElements) {
	double err = sqrt(error / totalDomainElements);
	if (err < 1)
		return true;
	else
		return false;
}

void DP45Storage::getDenseOutput(ProcessingUnit* pu, double timestep, double tetha, double* result) {
	pu->multiplyArrayByNumber(result, mTempStore1, getB1(tetha), mCount);
	pu->multiplyArrayByNumberAndSum(result, mTempStore3, getB3(tetha), mArg, mCount);
	pu->multiplyArrayByNumberAndSum(result, mTempStore4, getB4(tetha), mArg, mCount);
	pu->multiplyArrayByNumberAndSum(result, mTempStore5, getB5(tetha), mArg, mCount);
	pu->multiplyArrayByNumberAndSum(result, mTempStore6, getB6(tetha), mArg, mCount);
	pu->multiplyArrayByNumber(result, result, timestep, mCount);
	pu->sumArrays(result, result, mState, mCount);
}

void DP45Storage::print(ProcessingUnit* pu, int zCount, int yCount, int xCount, int cellSize) {
	printf("mState:\n");
	pu->printArray(mState, zCount, yCount, xCount, cellSize);

	printf("\nmTempStore1:\n");
	pu->printArray(mTempStore1, zCount, yCount, xCount, cellSize);

	printf("\nmTempStore2:\n");
	pu->printArray(mTempStore2, zCount, yCount, xCount, cellSize);

	printf("\nmTempStore3:\n");
	pu->printArray(mTempStore3, zCount, yCount, xCount, cellSize);

	printf("\nmTempStore4:\n");
	pu->printArray(mTempStore4, zCount, yCount, xCount, cellSize);

	printf("\nmTempStore5:\n");
	pu->printArray(mTempStore5, zCount, yCount, xCount, cellSize);

	printf("\nmTempStore6:\n");
	pu->printArray(mTempStore6, zCount, yCount, xCount, cellSize);

	printf("\nmTempStore7:\n");
	pu->printArray(mTempStore7, zCount, yCount, xCount, cellSize);

	printf("\nmArg:\n");
	pu->printArray(mArg, zCount, yCount, xCount, cellSize);
}
