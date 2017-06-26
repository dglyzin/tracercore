/*
 * state.cpp
 *
 *  Created on: 10 окт. 2016 г.
 *      Author: frolov
 */

#include "state.h"

using namespace std;

State::State(ProcessingUnit* _pu, NumericalMethod* _method, double** _blockCommonTempStorages, unsigned long long elementCount) {
	pu = _pu;
	method = _method;
	mBlockCommonTempStorages = _blockCommonTempStorages;
	mElementCount = elementCount;

	mState = pu->newDoubleArray(elementCount);

	int kStorageCount = method->getKStorageCount();

	//mKStorages = pu->newDoublePointerArray(kStorageCount);
	mKStorages = new double*[kStorageCount];
	for (int i = 0; i < kStorageCount; ++i) {
		mKStorages[i] = pu->newDoubleArray(mElementCount);
	}
}

State::~State() {
	/*pu->deleteDeviceSpecificArray(mState);

	 int kStorageCount = method->getKStorageCount();
	 for (int i = 0; i < kStorageCount; ++i) {
	 pu->deleteDeviceSpecificArray(mKStorages[i]);
	 }

	 pu->deleteDeviceSpecificArray(mKStorages);*/
	delete mKStorages;
}

/*double** State::getKStorages() {
 return mKStorages;
 }*/

/*double* State::getStorage(int storageNumber) {
 if (storageNumber < method->getKStorageCount())
 return mKStorages[storageNumber];
 else
 return NULL;
 }*/
void State::init(initfunc_fill_ptr_t* userInitFuncs, unsigned short int* initFuncNumber, int blockNumber, double time) {
	pu->initState(mState, userInitFuncs, initFuncNumber, blockNumber, 0.0);
}

double* State::getResultStorage(int stageNumber) {
	//return mStorages[method->getStorageNumberResult(stageNumber)];
	return method->getStorageResult(mState, mKStorages, mBlockCommonTempStorages, stageNumber);
}

double* State::getSourceStorage(int stageNumber) {
	//return mStorages[method->getStorageNumberSource(stageNumber)];
	return method->getStorageSource(mState, mKStorages, mBlockCommonTempStorages, stageNumber);
}

double* State::getState() {
	//return method->getStateStorage(mKStorages);
	return mState;
}

void State::getSubVolume(double* result, int zStart, int zStop, int yStart, int yStop, int xStart,
        int xStop, int yCount, int xCount, int cellSize){
	//printf("getting state subvolume z1 z2 y1 y2 x1 x2 yc xz cs: %d %d %d %d %d %d %d %d %d \n", zStart, zStop, yStart, yStop, xStart,
	//         xStop, yCount, xCount, cellSize);
	pu->getSubVolume(result, mState, zStart, zStop, yStart, yStop, xStart,
		         xStop, yCount, xCount, cellSize);

}

void State::setSubVolume(double* source, int zStart, int zStop, int yStart, int yStop, int xStart,
        int xStop, int yCount, int xCount, int cellSize){
	pu->setSubVolume(mState, source, zStart, zStop, yStart, yStop, xStart,
		         xStop, yCount, xCount, cellSize);

}

void State::prepareArgument(double timeStep, int stageNumber) {
	method->prepareArgument(pu, mState, mKStorages, mBlockCommonTempStorages, timeStep, stageNumber, mElementCount);
}

void State::computeDenseOutput(double timeStep, double theta, double* result) {
	method->computeDenseOutput(pu, mState, mKStorages, mBlockCommonTempStorages, timeStep, theta, result,
			mElementCount);
}

double State::computeStepError(double timeStep) {
	return method->computeStepError(pu, mState, mKStorages, mBlockCommonTempStorages, timeStep, mElementCount);
}

void State::confirmStep(double timeStep, State* nextStepState, ISmartCopy* sc) {
	method->confirmStep(pu, sc, &mState, mKStorages, &(nextStepState->mState), nextStepState->mKStorages,
			mBlockCommonTempStorages, timeStep, mElementCount);
}

void State::rejectStep(double timeStep) {
	method->rejectStep(pu, mState, mKStorages, mBlockCommonTempStorages, timeStep, mElementCount);
}

void State::saveGeneralStorage(char* path) {
	pu->saveArray(mState, mElementCount, path);
	//method->saveStateGeneralData(pu, mKStorages, path);
}

void State::saveAllStorage(char* path) {
	saveGeneralStorage(path);

	int kStorageCount = method->getKStorageCount();
	for (int i = 0; i < kStorageCount; ++i) {
		pu->saveArray(mKStorages[i], mElementCount, path);
	}
	//method->saveStateData(pu, mKStorages, path);
}

void State::saveStateForDrawDenseOutput(char* path, double timeStep, double theta) {
	double* result = pu->newDoubleArray(mElementCount);

	method->computeDenseOutput(pu, mState, mKStorages, mBlockCommonTempStorages, timeStep, theta, result,
			mElementCount);
	pu->saveArray(result, mElementCount, path);

	pu->deleteDeviceSpecificArray(result);
}

void State::loadGeneralStorage(ifstream& in) {
	pu->loadArray(mState, mElementCount, in);
	//method->loadStateGeneralData(pu, mKStorages, in);
}

void State::loadAllStorage(std::ifstream& in) {
	loadGeneralStorage(in);

	int kStorageCount = method->getKStorageCount();
	for (int i = 0; i < kStorageCount; ++i) {
		pu->loadArray(mKStorages[i], mElementCount, in);
	}
	//method->loadStateData(pu, mKStorages, in);
}

bool State::isNan() {
	return pu->isNan(mState, mElementCount);
}

void State::print(int zCount, int yCount, int xCount, int cellSize) {
	//printf("################################################################################");
	printf("\nState\n");
	printf("State address: %p\n", mState);

	int kStorageCount = method->getKStorageCount();
	for (int i = 0; i < kStorageCount; ++i) {
		printf("kStorage #%d address: %p\n", i, mKStorages[i]);
	}

	printf("\nState\n");
	pu->printArray(mState, zCount, yCount, xCount, cellSize);
	for (int i = 0; i < kStorageCount; ++i) {
		printf("\nkStorage #%d\n", i);
		pu->printArray(mKStorages[i], zCount, yCount, xCount, cellSize);
	}
	//printf("################################################################################");
	//printf("\n\n\n");
}
