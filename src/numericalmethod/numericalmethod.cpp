/*
 * numericalmethod.cpp
 *
 *  Created on: 13 окт. 2016 г.
 *      Author: frolov
 */

#include "numericalmethod.h"

NumericalMethod::NumericalMethod(double _aTol, double _rTol) {
	aTol = _aTol;
	rTol = _rTol;
}

NumericalMethod::~NumericalMethod() {
}

unsigned long long NumericalMethod::getMemorySizePerState(unsigned long long elementCount) {
	// state + temp storage count
	return (elementCount * SIZE_DOUBLE) * (1 + getKStorageCount());
}


int NumericalMethod::getArraysCountPerState() {
	// state + temp storage count
	return 2 + getKStorageCount();
}
