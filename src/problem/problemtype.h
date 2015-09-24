/*
 * problemtype.h
 *
 *  Created on: 24 сент. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_PROBLEM_PROBLEMTYPE_H_
#define SRC_PROBLEM_PROBLEMTYPE_H_

#include "../processingunit/processingunit.h"
#include "../stepstorage/eulerstorage.h"

class ProblemType {
public:
	ProblemType();
	virtual ~ProblemType();

	virtual double* getSource(int stage, double time) = 0;
	virtual double* getResult(int stage, double time) = 0;
};

#endif /* SRC_PROBLEM_PROBLEMTYPE_H_ */
