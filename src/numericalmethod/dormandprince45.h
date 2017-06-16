/*
 * dp45.h
 *
 *  Created on: 22 февр. 2017 г.
 *      Author: frolov
 */

#ifndef SRC_NUMERICALMETHOD_DP45_H_
#define SRC_NUMERICALMETHOD_DP45_H_

#include "numericalmethod.h"

class DormandPrince45: public NumericalMethod {
public:
	DormandPrince45(double _aTol, double _rTol);
	virtual ~DormandPrince45();

	int getStageCount();
	bool isFSAL();
	bool isErrorPermissible(double error, unsigned long long totalDomainElements);
	bool isVariableStep();
	double computeNewStep(double timeStep, double error, unsigned long long totalDomainElements);

	int getKStorageCount();
	int getCommonTempStorageCount();

	double* getStorageResult(double* state, double** kStorages, double** commonTempStorages, int stageNumber);
	double* getStorageSource(double* state, double** kStorages, double** commonTempStorages, int stageNumber);

	double getStageTimeStepCoefficient(int stageNumber);

	void prepareArgument(ProcessingUnit* pu, double* state, double** kStorages, double** commonTempStorages,
			double timeStep, int stageNumber, unsigned long long size);

	void confirmStep(ProcessingUnit* pu, ISmartCopy* sc, double** sourceState, double** sourceKStorages,
			double** destinationState, double** destinationKStorages, double** commonTempStorages, double timeStep,
			unsigned long long size);
	void rejectStep(ProcessingUnit* pu, double* state, double** kStorages, double** commonTempStorages, double timeStep,
			unsigned long long size);

	double computeStepError(ProcessingUnit* pu, double* state, double** kStorages, double** commonTempStorages,
			double timeStep, unsigned long long size);

	void computeDenseOutput(ProcessingUnit* pu, double* state, double** kStorages, double** commonTempStorages,
			double timeStep, double theta, double* result, unsigned long long size);

private:
	enum KSTORAGE {
		K1, K2, K3, K4, K5, K6, K7, KSTORAGE_COUNT
	};
	enum COMMON_TEMP_STROTAGE {
		ARG, TEMP1, TEMP2, COMMON_TEMP_STROTAGE_COUNT
	};

	static const double c2 = 0.2, c3 = 0.3, c4 = 0.8, c5 = 8.0 / 9.0;

	static const double a21 = 0.2, a31 = 3.0 / 40.0, a32 = 9.0 / 40.0;
	static const double a41 = 44.0 / 45.0, a42 = -(56.0 / 15.0), a43 = 32.0 / 9.0;
	static const double a51 = 19372.0 / 6561.0, a52 = -(25360.0 / 2187.0);
	static const double a53 = 64448.0 / 6561.0, a54 = -(212.0 / 729.0);
	static const double a61 = 9017.0 / 3168.0, a62 = -(355.0 / 33.0), a63 = 46732.0 / 5247.0;
	static const double a64 = 49.0 / 176.0, a65 = -(5103.0 / 18656.0);
	static const double a71 = 35.0 / 384.0, a73 = 500.0 / 1113.0, a74 = 125.0 / 192.0;
	static const double a75 = -(2187.0 / 6784.0), a76 = 11.0 / 84.0;
	static const double e1 = 71.0 / 57600.0, e3 = -(71.0 / 16695.0), e4 = 71.0 / 1920.0;
	static const double e5 = -(17253.0 / 339200.0), e6 = 22.0 / 525.0, e7 = -(1.0 / 40.0);
	static const double facmin = 0.5, facmax = 2, fac = 0.9;

	double getB1(double theta);
	double getB3(double theta);
	double getB4(double theta);
	double getB5(double theta);
	double getB6(double theta);
	double getB7(double theta);
};

#endif /* SRC_NUMERICALMETHOD_DP45_H_ */
