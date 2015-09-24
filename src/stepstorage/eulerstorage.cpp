/*
 * eulerstorage.cpp
 *
 *  Created on: 24 сент. 2015 г.
 *      Author: frolov
 */

#include "eulerstorage.h"

EulerStorage::EulerStorage() : StepStorage() {
}

EulerStorage::EulerStorage(ProcessingUnit* pc, int count, double _aTol, double _rTol) : StepStorage(pc, count, _aTol, _rTol) {

}

EulerStorage::~EulerStorage() {
	// TODO Auto-generated destructor stub
}

