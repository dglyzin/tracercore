/*
 * delayproblem.cpp
 *
 *  Created on: 2 мая 2017 г.
 *      Author: frolov2
 */

#include "delayproblem.h"

DelayProblem::DelayProblem(int _statesCount, int _delayCount, double* _delayValue) {
	statesCount = _statesCount;
	delayCount = _delayCount;

	delayValue = new double[delayCount];
	for (int i = 0; i < delayCount; ++i) {
		delayValue[i] = _delayValue[i];
	}

	delayStatesNumber = new int[delayCount];
	delayTheta = new double[delayCount];

	for (int i = 0; i < delayCount; ++i) {
		delayStatesNumber[i] = 0;
		delayTheta[i] = 0.0;
	}

	timeCountdown = new double[statesCount];

	for (int i = 0; i < statesCount; ++i) {
		timeCountdown[i] = 0.0;
	}

	currentStateNumber = 0;
}

DelayProblem::~DelayProblem() {
	delete delayStatesNumber;
	delete delayTheta;
	delete delayValue;
	delete timeCountdown;
}

void DelayProblem::computeStateNumberForDelay(double currentTime, double timeStep, double requiredTime, int index) {
	/*printf("\n\n************\n");
	for (int i = 0; i < statesCount; ++i) {
		printf("%.5f ", timeCountdown[i]);
		if ((i + 1) % 10 == 0)
			printf("\n");
	}
	printf("\n************\n");*/

	int stateNumber = delayStatesNumber[index];
	while (!(timeCountdown[(stateNumber + 1) % statesCount] > requiredTime)) {
		stateNumber = (stateNumber + 1) % statesCount;
	}

	delayStatesNumber[index] = stateNumber;

	/*printf("delay value: %f\ntime countdown: %f\nnew state index: %d\ncurrent time: %f\n", delayValue[index],
			timeCountdown[stateNumber], delayStatesNumber[index], currentTime);*/
}

void DelayProblem::computeTethaForDelay(double currentTime, double timeStep, double requiredTime, int index) {
	double denseTimeStep = timeCountdown[delayStatesNumber[index]]
			- timeCountdown[(delayStatesNumber[index] + 1) % statesCount];
	double timeShift = requiredTime - timeCountdown[delayStatesNumber[index]];

	delayTheta[index] = timeShift / denseTimeStep;
}

int DelayProblem::getCurrentStateNumber() {
	return currentStateNumber;
}

int DelayProblem::getNextStateNumber() {
	return (currentStateNumber + 1) % statesCount;
}

int DelayProblem::getStateCount() {
	return statesCount;
}

int DelayProblem::getDelayCount() {
	return delayCount;
}

double DelayProblem::getDelay(int delayNumber) {
	return delayValue[delayNumber];
}

void DelayProblem::computeStageData(double currentTime, double timeStep, double numericalMethodStageCoefficient) {
	for (int i = 0; i < delayCount; ++i) {
		double requiredTime = currentTime + numericalMethodStageCoefficient * timeStep - delayValue[i];
		computeStateNumberForDelay(currentTime, timeStep, requiredTime, i);
		computeTethaForDelay(currentTime, timeStep, requiredTime, i);
	}
}

void DelayProblem::confirmStep(double currentTime) {
	timeCountdown[stateNumberResult] = currentTime;
	currentStateNumber = (currentStateNumber + 1) % statesCount;
}

int DelayProblem::getStateNumberForDelay(int delayNumber) {
	return delayStatesNumber[delayNumber];
}

double DelayProblem::getTethaForDelay(int delayNumber) {
	return delayTheta[delayNumber];
}

void DelayProblem::loadData(std::ifstream& in, State** states) {
	states[getCurrentStateNumber()]->loadAllStorage(in);
}

void DelayProblem::saveData(char* path, State** states) {
	states[getCurrentStateNumber()]->saveAllStorage(path);
}

void DelayProblem::savaDataForDraw(char* path, State** states) {
	states[getCurrentStateNumber()]->saveGeneralStorage(path);
}

void DelayProblem::swapCopy(ProcessingUnit* pu, double** source, double** destination, int size) {
	pu->copyArray(*source, *destination, size);
}
