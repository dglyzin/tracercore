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
