/*
 * solver.h
 *
 *  Created on: Feb 12, 2015
 *      Author: dglyzin
 */

#ifndef SOLVER_H_
#define SOLVER_H_

enum SOLVER_IDX { EULER, RK4 };
int GetSolverStageCount(int solverIdx);


class Solver {
public:
    Solver();
    virtual ~Solver() { return; }
    void copyState(double* result);
    virtual void prepareStageData(int stage) { return; }
    virtual void confirmStep() { return; }
  	virtual void getStageArrays(double** result, double** source, double* factor, int stage, double timeStep) { return; }
  	double* getStatePtr(){ return mState;}

protected:
  	int     mCount;
  	double* mState;
};

Solver* GetCpuSolver(int solverIdx, int count);
Solver* GetGpuSolver(int solverIdx, int count);

class EulerSolver: public Solver{
public:
	EulerSolver(int _count);
	~EulerSolver();
    void prepareStageData(int stage) { return; }
    void getStageArrays(double** result, double** source, double* factor, int stage, double timeStep) { return; }
};




#endif /* SOLVER_H_ */
