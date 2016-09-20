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
	EulerStorage(ProcessingUnit* _pu, int count, double _aTol, double _rTol);
	virtual ~EulerStorage();

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

	void saveMTempStores(char* path);
	void loadMTempStores(std::ifstream& in);

	int getSizeChild(int elementCount);
};

#endif /* SRC_STEPSTORAGE_EULERSTORAGE_H_ */
