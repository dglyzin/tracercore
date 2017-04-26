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

void OrdinaryProblem::computeStateNumberForDelay(double currentTime, double timeStep) {
	return;
}

void OrdinaryProblem::computeTethaForDelay(double currentTime, double timeStep) {
	return;
}

int OrdinaryProblem::getStateCount() {
	return 1;
}

int OrdinaryProblem::getStateNumberResult(double currentTime) {
	return 0;
}

int OrdinaryProblem::getStateNumberSource(double currentTime) {
	return 0;
}

int OrdinaryProblem::getCurrentStateNumber() {
	return 0;
}

int OrdinaryProblem::getNextStateNumber() {
	return 0;
}

int OrdinaryProblem::getDelayCount() {
	return 0;
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
	pu->swapStorages(source, destination);
}
