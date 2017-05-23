/*
 * ordinaryproblem.cpp
 *
 *  Created on: 8 февр. 2017 г.
 *      Author: frolov
 */

#include "ordinaryproblem.h"

OrdinaryProblem::OrdinaryProblem() {
}

OrdinaryProblem::~OrdinaryProblem() {
}

int OrdinaryProblem::getCurrentStateNumber() {
	return 0;
}

int OrdinaryProblem::getNextStateNumber() {
	return 0;
}

int OrdinaryProblem::getStateCount() {
	return 1;
}

int OrdinaryProblem::getDelayCount() {
	return 0;
}

double OrdinaryProblem::getDelay(int delayNumber) {
	return 0;
}

void OrdinaryProblem::computeStageData(double currentTime, double timeStep, double numericalMethodStageCoefficient) {
	return;
}

void OrdinaryProblem::confirmStep(double currentTime) {
	return;
}

int OrdinaryProblem::getStateNumberForDelay(int delayNumber) {
	return 0;
}

double OrdinaryProblem::getTethaForDelay(int delayNumber) {
	return 0.0;
}

void OrdinaryProblem::loadData(std::ifstream& in, State** states) {
	states[getCurrentStateNumber()]->loadGeneralStorage(in);
}

void OrdinaryProblem::saveData(char* path, State** states) {
	states[getCurrentStateNumber()]->saveGeneralStorage(path);
}

void OrdinaryProblem::savaDataForDraw(char* path, State** states) {
	states[getCurrentStateNumber()]->saveGeneralStorage(path);
}

void OrdinaryProblem::swapCopy(ProcessingUnit* pu, double** source, double** destination, int size) {
	pu->swapArray(source, destination);
}
