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
	RK4Storage(ProcessingUnit* _pu, int count, double _aTol, double _rTol);
	virtual ~RK4Storage();

	double* getStageSource(int stage);
	double* getStageResult(int stage);

	double getStageTimeStep(int stage);

	void prepareArgument(int stage, double timestep);

	void confirmStep(double timestep);
	void rejectStep(double timestep);

	double getStepError(double timestep);

	bool isFSAL();
	bool isVariableStep();
	int getStageCount();

	double getNewStep(double timestep, double error, int totalDomainElements);
	bool isErrorPermissible(double error, int totalDomainElements);

	void getDenseOutput(double timestep, double tetha, double* result);

	void print(int zCount, int yCount, int xCount, int cellSize);

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

	void saveMTempStores(char* path);
	void loadMTempStores(std::ifstream& in);

	int getSizeChild(int elementCount);
};

#endif /* SRC_STEPSTORAGE_RK4STORAGE_H_ */
