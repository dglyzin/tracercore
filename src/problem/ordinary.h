/*
 * ordinary.h
 *
 *  Created on: 24 сент. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_PROBLEM_ORDINARY_H_
#define SRC_PROBLEM_ORDINARY_H_

#include "problemtype.h"

class Ordinary: public ProblemType {
public:
	Ordinary(ProcessingUnit* pu, int solverType, int count, double aTol, double rTol);
	virtual ~Ordinary();

private:
	StepStorage* stepStorage;
};

#endif /* SRC_PROBLEM_ORDINARY_H_ */
