/*
 * numericalmethod.h
 *
 *  Created on: 13 окт. 2016 г.
 *      Author: frolov
 */

#ifndef SRC_NUMERICALMETHOD_NUMERICALMETHOD_H_
#define SRC_NUMERICALMETHOD_NUMERICALMETHOD_H_

#include <math.h>
#include <cassert>

#include "../processingunit/processingunit.h"
#include "../problem/ismartcopy.h"

class NumericalMethod {
public:
	NumericalMethod(double _aTol, double _rTol);
	virtual ~NumericalMethod();

	virtual int getStageCount() = 0;
	virtual bool isFSAL() = 0;
	virtual bool isErrorPermissible(double error, unsigned long long totalDomainElements) = 0;
	virtual bool isVariableStep() = 0;
	virtual double computeNewStep(double timeStep, double error, unsigned long long totalDomainElements) = 0;

	virtual int getKStorageCount() = 0;
	virtual int getCommonTempStorageCount() = 0;

	virtual double* getStorageResult(double* state, double** kStorages, double** commonTempStorages,
			int stageNumber) = 0;
	virtual double* getStorageSource(double* state, double** kStorages, double** commonTempStorages,
			int stageNumber) = 0;

	//virtual double getStateStorage(double** kStorages) = 0;

	virtual double getStageTimeStepCoefficient(int stageNumber) = 0;

	virtual void prepareArgument(ProcessingUnit* pu, double* state, double** kStorages, double** commonTempStorages,
			double timeStep, int stageNumber, unsigned long long size) = 0;
	//virtual void prepareFSAL(ProcessingUnit* pu, double** source, double timeStep) = 0;

	virtual void confirmStep(ProcessingUnit* pu, ISmartCopy* sc, double** sourceState, double** sourceKStorages,
			double** destinationState, double** destinationKStorages, double** commonTemp, double timeStep,
			unsigned long long size) = 0;
	virtual void rejectStep(ProcessingUnit* pu, double* state, double** kStorages, double** commonTempStorages,
			double timeStep, unsigned long long size) = 0;

	virtual double computeStepError(ProcessingUnit* pu, double* state, double** kStorages, double** commonTempStorages,
			double timeStep, unsigned long long size) = 0;

	virtual void computeDenseOutput(ProcessingUnit* pu, double* state, double** kStorages, double** commonTempStorages,
			double timeStep, double theta, double* result, unsigned long long size) = 0;

	unsigned long long getMemorySizePerState(unsigned long long elementCount);

	//virtual bool isStateNan(ProcessingUnit* pu, double** kStorages) = 0;

	/*virtual void saveStateGeneralData(ProcessingUnit* pu, double** sourceStorages, char* path) = 0;
	 virtual void saveStateData(ProcessingUnit* pu, double** sourceStorages, char* path) = 0;

	 virtual void loadStateGeneralData(ProcessingUnit* pu, double** sourceStorages, std::ifstream& in) = 0;
	 virtual void loadStateData(ProcessingUnit* pu, double** sourceStorages, std::ifstream& in) = 0;*/

protected:
	double aTol;
	double rTol;
};

#endif /* SRC_NUMERICALMETHOD_NUMERICALMETHOD_H_ */
