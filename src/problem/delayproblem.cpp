/*
 * delayproblem.cpp
 *
 *  Created on: 2 мая 2017 г.
 *      Author: frolov2
 */

#include "delayproblem.h"

DelayProblem::DelayProblem(int _statesCount, int _delayCount, double* _delayValue) {
	stateNumberResult = 1;
	stateNumberSource = 0;

	statesCount = _statesCount;
	delayCount = _delayCount;

	delayValue = new double[delayCount];
	for (int i = 0; i < delayCount; ++i) {
		delayValue[i] = _delayValue[i];
	}

	delayStatesNumber = new int[delayCount];
	delayTheta = new double[delayCount];

	timeCountdown = new double[statesCount];

	currentStateNumber = 0;
}

DelayProblem::~DelayProblem() {
	delete delayStatesNumber;
	delete delayTheta;
	delete delayValue;
	delete timeCountdown;
}

void DelayProblem::computeStateNumberForDelay(double currentTime, double timeStep, double requiredTime, int index) {
	int stateNumber = delayStatesNumber[index];
	while (!(timeCountdown[(stateNumber + 1) % statesCount] > requiredTime)) {
		stateNumber = (stateNumber + 1) % statesCount;
	}

	delayStatesNumber[index] = stateNumber;
}

void DelayProblem::computeTethaForDelay(double currentTime, double timeStep, double requiredTime, int index) {
	double denseTimeStep = timeCountdown[delayStatesNumber[index]]
			- timeCountdown[(delayStatesNumber[index] + 1) % statesCount];
	double timeShift = requiredTime - timeCountdown[delayStatesNumber[index]];

	delayTheta[index] = timeShift / denseTimeStep;
}

int DelayProblem::getStateNumberResult(double currentTime) {
	return stateNumberResult;
}

int DelayProblem::getStateNumberSource(double currentTime) {
	return stateNumberSource;
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

void DelayProblem::computeStageData(double currentTime, double timeStep, double numericalMethodStagecoefficient) {
	for (int i = 0; i < delayCount; ++i) {
		double requiredTime = currentTime + numericalMethodStagecoefficient * timeStep - delayValue[i];

		computeStateNumberForDelay(currentTime, timeStep, requiredTime, i);
		computeTethaForDelay(currentTime, timeStep, requiredTime, i);
	}
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
