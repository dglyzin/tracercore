/*
 * rk4storage.h
 *
 *  Created on: 01 окт. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_STEPSTORAGE_RK4STORAGE_H_
#define SRC_STEPSTORAGE_RK4STORAGE_H_

#include "stepstorage.h"

class RK4Storage: public StepStorage {
public:
	RK4Storage();
	RK4Storage(ProcessingUnit* pu, int count, double _aTol, double _rTol);
	virtual ~RK4Storage();

	double* getStageSource(int stage);
	double* getStageResult(int stage);

	double getStageTimeStep(int stage);

	void prepareArgument(ProcessingUnit* pu, int stage, double timestep);

	void confirmStep(ProcessingUnit* pu, double timestep);
	void rejectStep(ProcessingUnit* pu, double timestep);

	double getStepError(ProcessingUnit* pu, double timestep);

	bool isFSAL();
	bool isVariableStep();
	int getStageCount();

	double getNewStep(double timestep, double error, int totalDomainElements);
	bool isErrorPermissible(double error, int totalDomainElements);

	void getDenseOutput(StepStorage* secondState, double* result);

	void print(ProcessingUnit* pu, int zCount, int yCount, int xCount, int cellSize);

private:
	double* mTempStore1;
	double* mTempStore2;
	double* mTempStore3;
	double* mTempStore4;
	double* mArg;

	static const double b1 = 1.0 / 6.0;
	static const double b2 = 1.0 / 3.0;
	static const double b3 = 1.0 / 3.0;
	static const double b4 = 1.0 / 6.0;

	void saveMTempStores(ProcessingUnit* pu, char* path);
	void loadMTempStores(ProcessingUnit* pu, std::ifstream& in);

	int sizeChild();
};

#endif /* SRC_STEPSTORAGE_RK4STORAGE_H_ */
