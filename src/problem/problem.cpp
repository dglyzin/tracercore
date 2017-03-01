/*
 * problem.cpp
 *
 *  Created on: 28 нояб. 2016 г.
 *      Author: frolov
 */

#include "../problem/problem.h"

Problem::Problem() : ISmartCopy() {
}

Problem::~Problem() {
}

void Problem::computeStageData(double currentTime, double timeStep) {
	computeStateNumberForDelay(currentTime, timeStep);
	computeTethaForDelay(currentTime, timeStep);
}
