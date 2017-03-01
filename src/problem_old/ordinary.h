/*
 * ordinary.h
 *
 *  Created on: 24 сент. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_PROBLEM_OLD_ORDINARY_H_
#define SRC_PROBLEM_OLD_ORDINARY_H_

#include "../problem_old/problemtype.h"

class Ordinary: public ProblemType {
public:
	Ordinary(ProcessingUnit* _pu, int solverType, int count, double aTol, double rTol);
	virtual ~Ordinary();

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
	void saveStateForDrawDenseOutput(char* path, double timestep, double tetha);
	void loadState(std::ifstream& in);

	bool isNan();

	void print(int zCount, int yCount, int xCount, int cellSize);

private:
	StepStorage* mStepStorage;
};

#endif /* SRC_PROBLEM_OLD_ORDINARY_H_ */
