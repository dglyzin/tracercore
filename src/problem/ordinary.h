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

	double** getSource(int stage);
	double* getResult(int stage);

	void prepareArgument(ProcessingUnit* pu, int stage, double timestep);

	double* getCurrentStateStageData(int stage);

	double getStepError(ProcessingUnit* pu, double timestep);

	void confirmStep(ProcessingUnit* pu, double timestep);
	void rejectStep(ProcessingUnit* pu, double timestep);

	void loadData(ProcessingUnit* pu, double* data);
	void getCurrentState(ProcessingUnit* pu, double* result);

	double* getCurrentStatePointer();

	void saveStateToDraw(ProcessingUnit* pu, char* path);
	void loadState(ProcessingUnit* pu, std::ifstream& in);

	bool isNan(ProcessingUnit* pu);

	void print(ProcessingUnit* pu, int zCount, int yCount, int xCount, int cellSize);

private:
	StepStorage* mStepStorage;
};

#endif /* SRC_PROBLEM_ORDINARY_H_ */
