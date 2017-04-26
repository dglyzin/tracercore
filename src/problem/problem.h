/*
 * problem.h
 *
 *  Created on: 28 нояб. 2016 г.
 *      Author: frolov
 */

#ifndef SRC_PROBLEM_OLD_PROBLEM_H_
#define SRC_PROBLEM_OLD_PROBLEM_H_

#include "ismartcopy.h"
#include "../processingunit/processingunit.h"
#include "../state.h"

class Problem : public ISmartCopy {
public:
	Problem();
	virtual ~Problem();

	virtual int getStateCount() = 0;

	virtual int getStateNumberResult(double currentTime) = 0;
	virtual int getStateNumberSource(double currentTime) = 0;
	virtual int getCurrentStateNumber() = 0;
	virtual int getNextStateNumber() = 0;

	virtual int getDelayCount() = 0;

	virtual int getStateNumberForDelay(int delayNumber) = 0;
	virtual double getTethaForDelay(int delayNumber) = 0;

	virtual void loadData(std::ifstream& in, State** states) = 0;
	virtual void saveData(char* path, State** states) = 0;
	virtual void savaDataForDraw(char* path, State** states) = 0;
	void saveStateForDrawDenseOutput(char* path, State** states, double timeStep, double theta);

	void computeStageData(double currentTime, double timeStep);

protected:
	virtual void computeStateNumberForDelay(double currentTime, double timeStep) = 0;
	virtual void computeTethaForDelay(double currentTime, double timeStep) = 0;
};

#endif /* SRC_PROBLEM_OLD_PROBLEM_H_ */
