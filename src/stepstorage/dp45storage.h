/*
 * dp45storage.h
 *
 *  Created on: 06 окт. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_STEPSTORAGE_DP45STORAGE_H_
#define SRC_STEPSTORAGE_DP45STORAGE_H_

#include "stepstorage.h"

class DP45Storage: public StepStorage {
public:
	DP45Storage();
	virtual ~DP45Storage();

    double* getStageSource(int stage);
    double* getStageResult(int stage);

    double getStageTimeStep(int stage);

    void prepareArgument(ProcessingUnit* pc, int stage, double timestep);

    void confirmStep(double timestep);
    void rejectStep(double timestep);

    double getStepError(double timeStep);

    bool isFSAL();
    bool isVariableStep();
    int getStageCount();

	double getNewStep(double timestep, double error, int totalDomainElements);
	bool isErrorPermissible(double error, int totalDomainElements);

	double* getDenseOutput(StepStorage* secondState);
};

#endif /* SRC_STEPSTORAGE_DP45STORAGE_H_ */
