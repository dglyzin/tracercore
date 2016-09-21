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
	Delay(ProcessingUnit* _pu, int solverType, int count, double aTol, double rTol, int _delayCount);
	virtual ~Delay();

	double** getSource(int stage);
	double* getResult(int stage);

	void prepareArgument(int stage, double timestep);

	double* getCurrentStateStageData(int stage);

	double getStepError(double timestep);

	void confirmStep(double timestep);
	void rejectStep(double timestep);

	void loadData(double* data);
	void getCurrentState(double* result);

	double* getCurrentStatePointer();

	void saveStateForDraw(char* path);
	void saveStateForLoad(char* path);
	void loadState(std::ifstream& in);

	bool isNan();

	void print(int zCount, int yCount, int xCount, int cellSize);

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
