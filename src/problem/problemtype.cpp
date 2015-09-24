/*
 * problemtype.cpp
 *
 *  Created on: 24 сент. 2015 г.
 *      Author: frolov
 */

#include "problemtype.h"

ProblemType::ProblemType() {
	// TODO Auto-generated constructor stub

}

ProblemType::~ProblemType() {
	// TODO Auto-generated destructor stub
}

StepStorage* ProblemType::createStageStorage(ProcessingUnit* pu, int solverType, int count, double aTol, double rTol) {
	switch (solverType) {
		case EULER:
			return EulerStorage(pu, count, aTol, rTol);
			break;
		default:
			break;
	}
}
