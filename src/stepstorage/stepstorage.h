/*
 * stepstorage.h
 *
 *  Created on: 24 сент. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_STEPSTORAGE_STEPSTORAGE_H_
#define SRC_STEPSTORAGE_STEPSTORAGE_H_

#include <math.h>

#include <cassert>

#include <fstream>

#include "../processingunit/processingunit.h"
#include "../enums.h"

class StepStorage {
public:
	StepStorage();
	StepStorage(ProcessingUnit* _pu, int count, double _aTol, double _rTol);
	virtual ~StepStorage();

	void copyState(double* result);

	void saveState(char* path);
	void loadState(std::ifstream& in);

	void saveStateWithTempStore(char* path);
	void loadStateWithTempStore(std::ifstream& in);

	void saveDenseOutput(char* path, double timestep, double tetha);

	double* getStatePointer();

	bool isNan();

	virtual double* getStageSource(int stage) = 0;
	virtual double* getStageResult(int stage) = 0;

	virtual double getStageTimeStep(int stage) = 0;

	virtual void prepareArgument(int stage, double timestep) = 0;

	virtual void confirmStep(double timestep) = 0;
	virtual void rejectStep(double timestep) = 0;

	virtual double getStepError(double timestep) = 0;

	virtual bool isFSAL() = 0;
	virtual bool isVariableStep() = 0;
	virtual int getStageCount() = 0;

	virtual double getNewStep(double timestep, double error, int totalDomainElements) = 0;
	virtual bool isErrorPermissible(double error, int totalDomainElements) = 0;

	virtual void getDenseOutput(double timestep, double tetha, double* result) = 0;

	int getSize(int elementCount);

	virtual void print(int zCount, int yCount, int xCount, int cellSize) = 0;

protected:
	ProcessingUnit* pu;

	int mCount;
	double* mState;

	double aTol;
	double rTol;

	void saveMState(char* path);
	void loadMState(std::ifstream& in);

	virtual void saveMTempStores(char* path) = 0;
	virtual void loadMTempStores(std::ifstream& in) = 0;

	virtual int getSizeChild(int elementCount) = 0;
};

#endif /* SRC_STEPSTORAGE_STEPSTORAGE_H_ */
