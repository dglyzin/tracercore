/*
 * solver.cpp
 *
 *  Created on: Feb 12, 2015
 *      Author: dglyzin
 */

#include "solver.h"

Solver::Solver() {
	mCount = 0;
	mState = NULL;
}

Solver::Solver(int _count){
	mCount = _count;
	mState = NULL;
}
