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
	ProblemType(ProcessingUnit* _pu);
	virtual ~ProblemType();

	virtual double** getSource(int stage) = 0;
	virtual double* getResult(int stage) = 0;

	virtual void prepareArgument(int stage, double timestep) = 0;

	virtual double* getCurrentStateStageData(int stage) = 0;

	virtual double getStepError(double timestep) = 0;

	virtual void confirmStep(double timestep) = 0;
	virtual void rejectStep(double timestep) = 0;

	virtual void loadData(double* data) = 0;
	virtual void getCurrentState(double* result) = 0;

	virtual double* getCurrentStatePointer() = 0;

	virtual void saveStateForDraw(char* path) = 0;
	virtual void saveStateForLoad(char* path) = 0;
	virtual void saveStateForDrawDenseOutput(char* path, double timestep, double tetha) = 0;
	virtual void loadState(std::ifstream& in) = 0;

	virtual bool isNan() = 0;

	virtual void print(int zCount, int yCount, int xCount, int cellSize) = 0;

protected:
	ProcessingUnit* pu;

	StepStorage* createStageStorage(int solverType, int count, double aTol, double rTol);

	double** mSourceStorage;
};

#endif /* SRC_PROBLEM_PROBLEMTYPE_H_ */
