/*
 * eulersolvergpu.h
 *
 *  Created on: 08 мая 2015 г.
 *      Author: frolov
 */

#ifndef SRC_SOLVERS_EULERSOLVERGPU_H_
#define SRC_SOLVERS_EULERSOLVERGPU_H_

#include "eulersolver.h"

class EulerSolverGpu: public EulerSolver {
public:
	EulerSolverGpu();
	virtual ~EulerSolverGpu();
};

#endif /* SRC_SOLVERS_EULERSOLVERGPU_H_ */
