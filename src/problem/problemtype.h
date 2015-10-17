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

	virtual void prepareArgument(ProcessingUnit* pu, int stage, double timestep) = 0;

	virtual double* getCurrentStateStageData(int stage) = 0;

	virtual double getStepError(ProcessingUnit* pu, double timestep) = 0;

protected:
	StepStorage* createStageStorage(ProcessingUnit* pu, int solverType, int count, double aTol, double rTol);
};

#endif /* SRC_PROBLEM_PROBLEMTYPE_H_ */
