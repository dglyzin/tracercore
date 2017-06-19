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
	State(ProcessingUnit* _pu, NumericalMethod* _method, double** _blockCommonTempStorages, unsigned long long elementCount);
	virtual ~State();

	//double* getStorage(int storageNumber);
	void init(initfunc_fill_ptr_t* userInitFuncs, unsigned short int* initFuncNumber, int blockNumber, double time);
	double* getResultStorage(int stageNumber);
	double* getSourceStorage(int stageNumber);

	double* getState();

	void prepareArgument(double timeStep, int stageNumber);
	void getSubVolume(double* result, int zStart, int zStop, int yStart, int yStop, int xStart,
	        int xStop, int yCount, int xCount, int cellSize);
	void setSubVolume(double* source, int zStart, int zStop, int yStart, int yStop, int xStart,
	        int xStop, int yCount, int xCount, int cellSize);

	void computeDenseOutput(double timeStep, double theta, double* result);

	double computeStepError(double timeStep);

	void confirmStep(double timeStep, State* nextStepState, ISmartCopy* sc);
	void rejectStep(double timeStep);

	void saveGeneralStorage(char* path);
	void saveAllStorage(char* path);
	void saveStateForDrawDenseOutput(char* path, double timeStep, double theta);

	void loadGeneralStorage(std::ifstream& in);
	void loadAllStorage(std::ifstream& in);

	bool isNan();

	void print(int zCount, int yCount, int xCount, int cellSize);

private:
	ProcessingUnit* pu;
	NumericalMethod* method;

	double* mState;
	double** mKStorages;
	double** mBlockCommonTempStorages;

	unsigned long long mElementCount;
};

#endif /* STATE_H_ */
