/*
 * problemtype.cpp
 *
 *  Created on: 24 сент. 2015 г.
 *      Author: frolov
 */

#include "problemtype.h"

ProblemType::ProblemType() {
	sourceStorage = NULL;
}

ProblemType::~ProblemType() {
	// TODO Auto-generated destructor stub
}

StepStorage* ProblemType::createStageStorage(ProcessingUnit* pu, int solverType, int count, double aTol, double rTol) {
	switch (solverType) {
		case EULER:
			return new EulerStorage(pu, count, aTol, rTol);
		case RK4:
			return new RK4Storage(pu, count, aTol, rTol);
		case DP45:
			return new DP45Storage(pu, count, aTol, rTol);
		default:
			return new EulerStorage(pu, count, aTol, rTol);
	}
}
