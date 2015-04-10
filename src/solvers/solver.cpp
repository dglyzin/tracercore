/*
 * solver.cpp
 *
 *  Created on: Feb 12, 2015
 *      Author: dglyzin
 */

#include <stdlib.h>
#include "solver.h"

int GetSolverStageCount(int solverIdx){
	if      (solverIdx == EULER)
		return 1;
	else if (solverIdx == RK4)
		return 4;
	else
		return -1;
}


Solver* GetCpuSolver(int solverIdx, int count){
	if      (solverIdx == EULER)
		return new EulerSolver(count);
	else if (solverIdx == RK4)
		return new EulerSolver(count);
	else
		return new EulerSolver(count);
}

Solver* GetGpuSolver(int solverIdx, int count){
	return NULL;
}



Solver::Solver(){

// printf("very strange action\n");
}

void Solver::copyState(double* result){
	for (int idx=0;idx<mCount;++idx)
		result[idx] = mState[idx];
}


EulerSolver::EulerSolver(int _count){
    mCount = _count;
    mState = new double[mCount];
    for (int i = 0; i < mCount; ++i)
        mState[i]= 0;
}

EulerSolver::~EulerSolver(){
    delete mState;
}
