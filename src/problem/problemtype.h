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
#include "../stepstorage/rk4storage.h"
#include "../stepstorage/dp45storage.h"

class ProblemType {
public:
	ProblemType();
	virtual ~ProblemType();

	virtual double** getSource(int stage) = 0;
	virtual double* getResult(int stage) = 0;

	virtual void prepareArgument(ProcessingUnit* pu, int stage, double timestep) = 0;

	virtual double* getCurrentStateStageData(int stage) = 0;

	virtual double getStepError(ProcessingUnit* pu, double timestep) = 0;

	virtual void confirmStep(ProcessingUnit* pu, double timestep) = 0;
	virtual void rejectStep(ProcessingUnit* pu, double timestep) = 0;

	virtual void loadData(ProcessingUnit* pu, double* data) = 0;
	virtual void getCurrentState(ProcessingUnit* pu, double* result) = 0;

	virtual double* getCurrentStatePointer() = 0;

	virtual void saveState(ProcessingUnit* pu, std::ofstream& out) = 0;
	virtual void loadState(ProcessingUnit* pu, std::ifstream& in) = 0;

protected:
	StepStorage* createStageStorage(ProcessingUnit* pu, int solverType, int count, double aTol, double rTol);

	double** mSourceStorage;
};

#endif /* SRC_PROBLEM_PROBLEMTYPE_H_ */
