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
    virtual void confirmStep() { return; }
    virtual double* getStageSource(int stage) { return NULL; }
  	virtual double* getStageResult(int stage) { return NULL; }
  	virtual double  getStageFactor(int stage, double timeStep) { return 0; }
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
	double* getStageSource(int stage);
	double* getStageResult(int stage);
	double  getStageFactor(int stage, double timeStep);
	void confirmStep();
private:
    double* mTempStore1;
};




#endif /* SOLVER_H_ */
