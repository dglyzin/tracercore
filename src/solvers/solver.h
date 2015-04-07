/*
 * solver.h
 *
 *  Created on: Feb 12, 2015
 *      Author: dglyzin
 */

#ifndef SOLVER_H_
#define SOLVER_H_

class Solver {
public:
    Solver();
    virtual void prepareStageData(int stage) { return; }
  	virtual void computeOneStageBorder(double time, double* param, int stage) { return; }
  	virtual void computeOneStageCenter(double time, double* param, int stage) { return; }



};



#endif /* SOLVER_H_ */
