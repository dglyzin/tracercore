/*
 * delayproblem.cpp
 *
 *  Created on: 2 мая 2017 г.
 *      Author: frolov2
 */

#include "delayproblem.h"

DelayProblem::DelayProblem(int _delayCount, int _statesCount) {
	stateNumberResult = 1;
	stateNumberSource = 0;

	delayCount = _delayCount;
	statesCount = _statesCount;

	delayStatesNumber = new int[delayCount];
	delayTheta = new double[delayCount];

	timeCountdown = new double[delayCount];
}

DelayProblem::~DelayProblem() {
	delete delayStatesNumber;
	delete delayTheta;
	delete timeCountdown;
}

void DelayProblem::computeStateNumberForDelay(double currentTime, double timeStep,
		double numericalMethodStagecoefficient) {

}

void DelayProblem::computeTethaForDelay(double currentTime, double timeStep, double numericalMethodStagecoefficient) {

}

int DelayProblem::getStateNumberResult(double currentTime) {
	return stateNumberResult;
}

int DelayProblem::getStateNumberSource(double currentTime) {
	return stateNumberSource;
}

int DelayProblem::getCurrentStateNumber() {
	return 0;
}

int DelayProblem::getNextStateNumber() {
	return 0;
}

int DelayProblem::getStateCount() {
	return statesCount;
}

int DelayProblem::getDelayCount() {
	return delayCount;
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
