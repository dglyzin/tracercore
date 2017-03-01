/*
 * state.h
 *
 *  Created on: 10 окт. 2016 г.
 *      Author: frolov
 */

#ifndef STATE_H_
#define STATE_H_

#include "numericalmethod/numericalmethod.h"

class State {
public:
	State(ProcessingUnit* _pu, NumericalMethod* _method, double** _blockCommonTempStorages, int elementCount);
	virtual ~State();

	//double* getStorage(int storageNumber);
	double* getResultStorage(int stageNumber);
	double* getSourceStorage(int stageNumber);

	double* getState();

	void prepareArgument(double timeStep, int stageNumber);

	void computeDenseOutput(double timeStep, double theta, double* result);

	double computeStepError(double timeStep);

	void confirmStep(double timeStep, State* nextStepState, ISmartCopy* sc);
	void rejectStep(double timeStep);

	void saveGeneralStorage(char* path);
	void saveAllStorage(char* path);

	void loadGeneralStorage(std::ifstream& in);
	void loadAllStorage(std::ifstream& in);

	bool isNan();

	void print();

private:
	ProcessingUnit* pu;
	NumericalMethod* method;

	double* mState;
	double** mKStorages;
	double** mBlockCommonTempStorages;

	int mElementCount;
};

#endif /* STATE_H_ */
