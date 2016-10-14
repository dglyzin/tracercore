/*
 * state.cpp
 *
 *  Created on: 10 окт. 2016 г.
 *      Author: frolov
 */

#include "state.h"

State::State(ProcessingUnit* _pu, int storeCount, int elementCount) {
	pu = _pu;

	mStoreCount = storeCount;
	mElementCount = elementCount;

	mStores = pu->newDoublePointerArray(mStoreCount);
	for (int i = 0; i < storeCount; ++i) {
		mStores[i] = pu->newDoubleArray(mElementCount);
	}
}

State::~State() {
	for (int i = 0; i < mStoreCount; ++i) {
		pu->deleteDeviceSpecificArray(mStores[i]);
	}

	pu->deleteDeviceSpecificArray(mStores);
}

double* State::getStore(int storeNumber) {
	return mStores[storeNumber];
}

void State::saveGeneralStore(char* path) {
	pu->saveArray(mStores[0], mElementCount, path);
}

void State::saveAllStores(char* path) {
	for (int i = 0; i < mStoreCount; ++i) {
		pu->saveArray(mStores[i], mElementCount, path);
	}
}
