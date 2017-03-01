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

	int getStateCount();

	int getStateNumberResult(double currentTime);
	int getStateNumberSource(double currentTime);
	int getCurrentStateNumber();
	int getNextStateNumber();

	int getDelayCount();

	int getStateNumberForDelay(int delayNumber);
	double getTethaForDelay(int delayNumber);

	void loadData(std::ifstream& in, State** states);
	void saveData(char* path, State** states);
	void savaDataForDraw(char* path, State** states);

	void swapCopy(ProcessingUnit* pu, double** source, double** destination, int size);

protected:
	void computeStateNumberForDelay(double currentTime, double timeStep);
	void computeTethaForDelay(double currentTime, double timeStep);
};

#endif /* SRC_PROBLEM_ORDINARYPROBLEM_H_ */
