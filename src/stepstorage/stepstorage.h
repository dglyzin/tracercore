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
	StepStorage(ProcessingUnit* pu, int count, double _aTol, double _rTol);
	virtual ~StepStorage();

    void copyState(ProcessingUnit* pu, double* result);

    void saveState(ProcessingUnit* pu, char* path);
    void loadState(ProcessingUnit* pu, std::ifstream& in);

	void saveStateWithTempStore(ProcessingUnit* pu, char* path);
    void loadStateWithTempStore(ProcessingUnit* pu, std::ifstream& in);

    double* getStatePointer();

    virtual double* getStageSource(int stage) = 0;
    virtual double* getStageResult(int stage) = 0;

    virtual double getStageTimeStep(int stage) = 0;

    virtual void prepareArgument(ProcessingUnit* pu, int stage, double timestep) = 0;

    virtual void confirmStep(ProcessingUnit* pu, double timestep) = 0;
    virtual void rejectStep(ProcessingUnit* pu, double timestep) = 0;

    virtual double getStepError(ProcessingUnit* pu, double timestep) = 0;

    virtual bool isFSAL() = 0;
    virtual bool isVariableStep() = 0;
    virtual int getStageCount() = 0;

	virtual double getNewStep(double timestep, double error, int totalDomainElements) = 0;
	virtual bool isErrorPermissible(double error, int totalDomainElements) = 0;

	virtual void getDenseOutput(StepStorage* secondState, double* result) = 0;

protected:
  	int     mCount;
  	double* mState;

  	double aTol;
  	double rTol;

  	void saveMState(ProcessingUnit* pu, char* path);
  	void loadMState(ProcessingUnit* pu, std::ifstream& in);

  	virtual void saveMTempStores(ProcessingUnit* pu, char* path) = 0;
  	virtual void loadMTempStores(ProcessingUnit* pu, std::ifstream& in) = 0;
};

#endif /* SRC_STEPSTORAGE_STEPSTORAGE_H_ */
