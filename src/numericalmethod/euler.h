/*
 * euler.h
 *
 *  Created on: 13 февр. 2017 г.
 *      Author: frolov
 */

#ifndef SRC_NUMERICALMETHOD_EULER_H_
#define SRC_NUMERICALMETHOD_EULER_H_

#include "numericalmethod.h"

class Euler: public NumericalMethod {
public:
	Euler(double _aTol, double _rTol);
	virtual ~Euler();

	int getStageCount();
	bool isFSAL();
	bool isErrorPermissible(double error, int totalDomainElements);
	bool isVariableStep();
	double computeNewStep(double timeStep, double error, int totalDomainElements);

	int getMemorySizePerState(int elementCount);

	int getKStorageCount();
	int getCommonTempStorageCount();

	double* getStorageResult(double* state, double** kStorages, double** commonTempStorages, int stageNumber);
	double* getStorageSource(double* state, double** kStorages, double** commonTempStorages, int stageNumber);

	double getStageTimeStepCoefficient(int stageNumber);

	void prepareArgument(ProcessingUnit* pu, double* state, double** kStorages, double** commonTempStorages,
			double timeStep, int stageNumber, int size);

	void confirmStep(ProcessingUnit* pu, ISmartCopy* sc, double** sourceState, double** sourceKStorages,
			double** destinationState, double** destinationKStorages, double** commonTemp, double timeStep, int size);
	void rejectStep(ProcessingUnit* pu, double* state, double** kStorages, double** commonTempStorages, double timeStep,
			int size);

	double computeStepError(ProcessingUnit* pu, double** kStorages, double** commonTempStorages, double timeStep,
			int size);

	void computeDenseOutput(ProcessingUnit* pu, double* state, double** kStorages, double timeStep, double theta,
			double* result, int size);

private:
	enum KSTORAGE {
		K1, KSTORAGE_COUNT
	};
};

#endif /* SRC_NUMERICALMETHOD_EULER_H_ */
