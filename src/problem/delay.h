/*
 * delay.h
 *
 *  Created on: 16 дек. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_PROBLEM_DELAY_H_
#define SRC_PROBLEM_DELAY_H_

#include "problemtype.h"

class Delay: public ProblemType {
public:
	Delay(ProcessingUnit* pu, int solverType, int count, double aTol, double rTol, int _delayCount);
	virtual ~Delay();

	double** getSource(int stage, double time);
	double* getResult(int stage, double time);

	void prepareArgument(ProcessingUnit* pu, int stage, double timestep);

	double* getCurrentStateStageData(int stage);

	double getStepError(ProcessingUnit* pu, double timestep);

	void confirmStep(ProcessingUnit* pu, double timestep);
	void rejectStep(ProcessingUnit* pu, double timestep);

	void loadData(ProcessingUnit* pu, double* data);
	void getCurrentState(ProcessingUnit* pu, double* result);

	double* getCurrentStatePointer();

private:
	StepStorage** mStepStorage;

	int delayCount;

	int maxStorageCount;

	int currentStorageNumber;


	int getSourceStorageNumber(double time);
	int getSourceStorageNumberDelay(double time, int delayNumber);
	int getSourceStorageNumberDelayForDenseOutput(double time, int delayNumber);
	int getResultStorageNumber();
};

#endif /* SRC_PROBLEM_DELAY_H_ */
