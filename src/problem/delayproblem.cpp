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
	double denseTimeStep = timeCountdown[(delayStatesNumber[index] + 1) % statesCount]
			- timeCountdown[delayStatesNumber[index]];
	double timeShift = requiredTime - timeCountdown[delayStatesNumber[index]];

	delayTheta[index] = timeShift / denseTimeStep;
}

int DelayProblem::getMaxDelayIndex() {
	int maxDelayIndex = 0;

	for (int i = 1; i < delayCount; ++i) {
		if(delayValue[i] > delayValue[maxDelayIndex]) {
			maxDelayIndex = i;
		}
	}

	return maxDelayIndex;
}

int DelayProblem::getStateNumberMaxDelay() {
	return delayStatesNumber[getMaxDelayIndex()];
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
		/*if (requiredTime > 0)
			printf("rt: %f, theta: %f, ct: %f, tc: %f,  sn: %d\n", requiredTime, delayTheta[i], currentTime,
					timeCountdown[delayStatesNumber[i]], delayStatesNumber[i]);*/
		/*if(delayTheta[i] > 1 || delayTheta[i] < 0)
			printf("***\n");*/
	}
}

void DelayProblem::confirmStep(double currentTime) {
	timeCountdown[currentStateNumber] = currentTime;
	currentStateNumber = (currentStateNumber + 1) % statesCount;
}

int DelayProblem::getStateNumberForDelay(int delayNumber) {
	return delayStatesNumber[delayNumber];
}

double DelayProblem::getTethaForDelay(int delayNumber) {
	return delayTheta[delayNumber];
}

void DelayProblem::load(std::ifstream& in) {
	in.read((char*) &currentStateNumber, SIZE_DOUBLE);

	for (int i = 0; i < delayCount; ++i) {
		in.read((char*) &delayStatesNumber[i], SIZE_INT);
	}

	int statenumberMaxDelay = getStateNumberMaxDelay();

	for (int i = statenumberMaxDelay; i != currentStateNumber; i = (i + 1) % statesCount) {
		in.read((char*) &timeCountdown[i], SIZE_DOUBLE);
	}
}

void DelayProblem::save(char* path) {
	ofstream out;
	out.open(path, ios::binary | ios::app);

	out.write((char*) &currentStateNumber, SIZE_DOUBLE);

	for (int i = 0; i < delayCount; ++i) {
		out.write((char*) &delayStatesNumber[i], SIZE_INT);
	}

	int stateNumberMaxDelay = getStateNumberMaxDelay();

	for (int i = stateNumberMaxDelay; i != currentStateNumber; i = (i + 1) % statesCount) {
		out.write((char*) &timeCountdown[i], SIZE_DOUBLE);
	}

	out.close();
}

void DelayProblem::loadData(std::ifstream& in, State** states) {
	//states[getCurrentStateNumber()]->loadAllStorage(in);
	int statenumberMaxDelay = getStateNumberMaxDelay();
	for (int i = statenumberMaxDelay; i != currentStateNumber; i = (i + 1) % statesCount) {
		states[i]->loadAllStorage(in);
	}
}

void DelayProblem::saveData(char* path, State** states) {
	//states[getCurrentStateNumber()]->saveAllStorage(path);
	int statenumberMaxDelay = getStateNumberMaxDelay();
	for (int i = statenumberMaxDelay; i != currentStateNumber; i = (i + 1) % statesCount) {
		states[i]->saveAllStorage(path);
	}
}

void DelayProblem::savaDataForDraw(char* path, State** states) {
	states[getCurrentStateNumber()]->saveGeneralStorage(path);
}

void DelayProblem::swapCopy(ProcessingUnit* pu, double** source, double** destination, int size) {
	pu->copyArray(*source, *destination, size);
}
