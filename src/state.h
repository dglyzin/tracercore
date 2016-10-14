/*
 * state.h
 *
 *  Created on: 10 окт. 2016 г.
 *      Author: frolov
 */

#ifndef STATE_H_
#define STATE_H_

#include "processingunit/processingunit.h"

class State {
public:
	State(ProcessingUnit* _pu, int storeCount, int elementCount);
	virtual ~State();

	double* getStore(int storeNumber);

	void saveGeneralStore(char* path);
	void saveAllStores(char* path);

private:
	ProcessingUnit* pu;

	double** mStores;

	int mStoreCount;
	int mElementCount;
};

#endif /* STATE_H_ */
