/*
 * rungekutta.h
 *
 *  Created on: 14 мар. 2017 г.
 *      Author: frolov
 */

#ifndef SRC_NUMERICALMETHOD_RUNGEKUTTA4_H_
#define SRC_NUMERICALMETHOD_RUNGEKUTTA4_H_

#include "numericalmethod.h"

class RungeKutta4: public NumericalMethod {
public:
	RungeKutta4(double _aTol, double _rTol);
	virtual ~RungeKutta4();

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

	double computeStepError(ProcessingUnit* pu, double* state, double** kStorages, double** commonTempStorages,
			double timeStep, int size);

	void computeDenseOutput(ProcessingUnit* pu, double* state, double** kStorages, double timeStep, double theta,
			double* result, int size);

private:
	enum KSTORAGE {
		K1, K2, K3, K4, KSTORAGE_COUNT
	};

	enum COMMON_TEMP_STROTAGE {
		ARG, COMMON_TEMP_STROTAGE_COUNT
	};

	static const double b1 = 1.0 / 6.0;
	static const double b2 = 1.0 / 3.0;
	static const double b3 = 1.0 / 3.0;
	static const double b4 = 1.0 / 6.0;
};

#endif /* SRC_NUMERICALMETHOD_RUNGEKUTTA4_H_ */
