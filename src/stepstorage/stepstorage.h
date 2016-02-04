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

#include "../processingunit/processingunit.h"
#include "../enums.h"

class StepStorage {
public:
	StepStorage();
	StepStorage(ProcessingUnit* pu, int count, double _aTol, double _rTol);
	virtual ~StepStorage();

    void copyState(ProcessingUnit* pu, double* result);

    void save(ProcessingUnit* pu, char* path);
    void load(ProcessingUnit* pu, char* path);

    double* getStatePointer() { return mState; }

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
};

#endif /* SRC_STEPSTORAGE_STEPSTORAGE_H_ */
