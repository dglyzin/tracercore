/*
 * eulerstorage.cpp
 *
 *  Created on: 24 сент. 2015 г.
 *      Author: frolov
 */

#include "eulerstorage.h"

EulerStorage::EulerStorage() : StepStorage() {
	mTempStore1 = NULL;
}

EulerStorage::EulerStorage(ProcessingUnit* pu, int count, double _aTol, double _rTol) : StepStorage(pc, count, _aTol, _rTol) {
	mTempStore1 = pu->newDoubleArray(mCount);
}

EulerStorage::~EulerStorage() {
	// TODO Auto-generated destructor stub
}
