/*
 * eulerstorage.h
 *
 *  Created on: 24 сент. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_STEPSTORAGE_EULERSTORAGE_H_
#define SRC_STEPSTORAGE_EULERSTORAGE_H_

#include "stepstorage.h"

class EulerStorage: public StepStorage {
public:
	EulerStorage();
	EulerStorage(ProcessingUnit* pc, int count, double _aTol, double _rTol);
	virtual ~EulerStorage();

    double* getStageSource(int stage);
    double* getStageResult(int stage);

    double getStageTimeStep(int stage);

    void prepareArgument(int stage, double timeStep);

    void confirmStep(double timestep);
    void rejectStep(double timestep);

    double getStepError(double timeStep);

    bool isFSAL();
    bool isVariableStep();
    int getStageCount();

	double getNewStep(double timeStep, double error, int totalDomainElements);
	bool isErrorPermissible(double error, int totalDomainElements);

	double* getDenseOutput(Solver* secondState);

private:
	double* mTempStore1;
};

#endif /* SRC_STEPSTORAGE_EULERSTORAGE_H_ */
