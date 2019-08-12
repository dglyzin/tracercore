/*
 * ordinaryproblem.h
 *
 *  Created on: 8 февр. 2017 г.
 *      Author: frolov
 */

#ifndef SRC_PROBLEM_ORDINARYPROBLEM_H_
#define SRC_PROBLEM_ORDINARYPROBLEM_H_

#include "problem.h"

class OrdinaryProblem: public Problem {
public:
	OrdinaryProblem();
	virtual ~OrdinaryProblem();

	int getCurrentStateNumber();
	int getNextStateNumber();

	int getStateCount();

	int getDelayCount();
	double getDelay(int delayNumber);

	void computeStageData(double currentTime, double timeStep, double numericalMethodStageCoefficient);

	void confirmStep(double currentTime);

	int getStateNumberForDelay(int delayNumber);
	double getTethaForDelay(int delayNumber);
	double getTimeStepForDelay(int delayNumber);

	void load(std::ifstream& in);
	void save(char* path);

	void loadData(std::ifstream& in, State** states);
	void saveData(char* path, State** states);
	void savaDataForDraw(char* path, State** states);

	void swapCopy(ProcessingUnit* pu, double** source, double** destination, unsigned long long size);
};

#endif /* SRC_PROBLEM_ORDINARYPROBLEM_H_ */
