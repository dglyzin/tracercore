/*
 * problem.cpp
 *
 *  Created on: 28 нояб. 2016 г.
 *      Author: frolov
 */

#include "../problem/problem.h"

Problem::Problem() :
		ISmartCopy() {
}

Problem::~Problem() {
}

void Problem::computeStageData(double currentTime, double timeStep, double numericalMethodStagecoefficient) {
	computeStateNumberForDelay(currentTime, timeStep, numericalMethodStagecoefficient);
	computeTethaForDelay(currentTime, timeStep, numericalMethodStagecoefficient);
}

void Problem::saveStateForDrawDenseOutput(char* path, State** states, double timeStep, double theta) {
	states[getCurrentStateNumber()]->saveStateForDrawDenseOutput(path, timeStep, theta);
}
