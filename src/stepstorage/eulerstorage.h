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
	EulerStorage(ProcessingUnit* pu, int count, double _aTol, double _rTol);
	virtual ~EulerStorage();

    double* getStageSource(int stage);
    double* getStageResult(int stage);

    double getStageTimeStep(int stage);

    void prepareArgument(ProcessingUnit* pu, int stage, double timestep);

    void confirmStep(ProcessingUnit* pu, double timestep);
    void rejectStep(ProcessingUnit* pu, double timestep);

    double getStepError(double timeStep);

    bool isFSAL();
    bool isVariableStep();
    int getStageCount();

	double getNewStep(double timestep, double error, int totalDomainElements);
	bool isErrorPermissible(double error, int totalDomainElements);

	double* getDenseOutput(StepStorage* secondState);

private:
	double* mTempStore1;
};

#endif /* SRC_STEPSTORAGE_EULERSTORAGE_H_ */
