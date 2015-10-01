/*
 * stepstorage.h
 *
 *  Created on: 24 сент. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_STEPSTORAGE_STEPSTORAGE_H_
#define SRC_STEPSTORAGE_STEPSTORAGE_H_

#include <stdlib.h>

#include "../processingunit/processingunit.h"
#include "../enums.h"

class StepStorage {
public:
	StepStorage();
	StepStorage(ProcessingUnit* pu, int count, double _aTol, double _rTol);
	virtual ~StepStorage();

    void copyState(ProcessingUnit* pc, double* result);
    void loadState(ProcessingUnit* pc, double* data);

    double* getStatePtr() { return mState;}

    virtual double* getStageSource(int stage) = 0;
    virtual double* getStageResult(int stage) = 0;

    virtual double getStageTimeStep(int stage) = 0;

    virtual void prepareArgument(ProcessingUnit* pc, int stage, double timestep) = 0;

    virtual void confirmStep(double timestep) = 0;
    virtual void rejectStep(double timestep) = 0;

    virtual double getStepError(double timeStep) = 0;

    virtual bool isFSAL() = 0;
    virtual bool isVariableStep() = 0;
    virtual int getStageCount() = 0;

	virtual double getNewStep(double timeStep, double error, int totalDomainElements) = 0;
	virtual bool isErrorPermissible(double error, int totalDomainElements) = 0;

	virtual double* getDenseOutput(StepStorage* secondState) = 0;

protected:
  	int     mCount;
  	double* mState;

  	double aTol;
  	double rTol;
};

#endif /* SRC_STEPSTORAGE_STEPSTORAGE_H_ */
