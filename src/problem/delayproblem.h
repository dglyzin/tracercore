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
	DelayProblem(int _statesCount, int _delayCount, double* _delayValue);
	virtual ~DelayProblem();

	int getCurrentStateNumber();
	int getNextStateNumber();

	int getStateCount();

	int getDelayCount();
	double getDelay(int delayNumber);

	void computeStageData(double currentTime, double timeStep, double numericalMethodStageCoefficient);

	void confirmStep(double currentTime);

	int getStateNumberForDelay(int delayNumber);
	double getTethaForDelay(int delayNumber);

	void load(std::ifstream& in);
	void save(char* path);

	void loadData(std::ifstream& in, State** states);
	void saveData(char* path, State** states);
	void savaDataForDraw(char* path, State** states);

	void swapCopy(ProcessingUnit* pu, double** source, double** destination,
			int size);

private:
	int delayCount;
	double* delayValue;

	int statesCount;

	int* delayStatesNumber;
	double* delayTheta;

	double* timeCountdown;

	int currentStateNumber;

	void computeStateNumberForDelay(double currentTime, double timeStep, double requiredTime, int index);
	void computeTethaForDelay(double currentTime, double timeStep, double requiredTime, int index);

	int getMaxDelayIndex();
	int getStateNumberMaxDelay();
};

#endif /* SRC_PROBLEM_DELAYPROBLEM_H_ */
