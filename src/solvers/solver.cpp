/*
 * solver.cpp
 *
 *  Created on: Feb 12, 2015
 *      Author: dglyzin
 */

#include "solver.h"

int GetSolverStageCount1(int solverIdx){
	if      (solverIdx == EULER)
		return 1;
	else if (solverIdx == RK4)
		return 4;
	else
		return -1;
}



Solver::Solver(){

// printf("very strange action\n");
}

