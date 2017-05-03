/*
 * delayproblem.h
 *
 *  Created on: 2 мая 2017 г.
 *      Author: frolov2
 */

#ifndef SRC_PROBLEM_DELAYPROBLEM_H_
#define SRC_PROBLEM_DELAYPROBLEM_H_

#include "problem.h"

class DelayProblem: public Problem {
public:
	DelayProblem(int _delayCount, int _statesCount);
	virtual ~DelayProblem();

	int getStateNumberResult(double currentTime);
	int getStateNumberSource(double currentTime);
	int getCurrentStateNumber();
	int getNextStateNumber();

	int getStateCount();

	int getDelayCount();

	int getStateNumberForDelay(int delayNumber);
	double getTethaForDelay(int delayNumber);

	void loadData(std::ifstream& in, State** states);
	void saveData(char* path, State** states);
	void savaDataForDraw(char* path, State** states);

	void swapCopy(ProcessingUnit* pu, double** source, double** destination,
			int size);

protected:
	void computeStateNumberForDelay(double currentTime, double timeStep, double numericalMethodStagecoefficient);
	void computeTethaForDelay(double currentTime, double timeStep, double numericalMethodStagecoefficient);

private:
	int delayCount;

	int statesCount;

	int stateNumberResult;
	int stateNumberSource;

	int* delayStatesNumber;
	double* delayTheta;

	double* timeCountdown;
};

#endif /* SRC_PROBLEM_DELAYPROBLEM_H_ */
