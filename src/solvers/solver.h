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
    virtual ~Solver() { return; }
    void copyState(double* result);
    double* getStatePtr(){ return mState;}

    virtual double* getStageSource(int stage) { return NULL; }
    virtual double* getStageResult(int stage) { return NULL; }
    virtual void prepareArgument(int stage) { return; }
    virtual void confirmStep() { return; }
    virtual double getStepError() { return 0.0; }


protected:
  	int     mCount;
  	double* mState;
};

class SolverInfo{
public:
	SolverInfo();
	virtual ~SolverInfo() { return; }
	int isFSAL() { return mIsFSAL; } //gets a solver 'first same as last' property
	                                   //to prepare data on first step
	int isVariableStep() { return mVariableStep; }
	int getStageCount() { return mStageCount; }

	virtual double getNewStep(double timeStep, double error) { return timeStep; }
	virtual int isErrorOK(double error) { return 1; }

protected:
	int mIsFSAL;
	int mVariableStep;
	int mStageCount;
};

/*Solver* GetCpuSolver(int solverIdx, int count);
Solver* GetGpuSolver(int solverIdx, int count);*/

class EulerSolver: public Solver{
public:
	EulerSolver(int _count);
	~EulerSolver();
	double* getStageSource(int stage);
	double* getStageResult(int stage);
	void prepareArgument(int stage,double timeStep);
	void confirmStep();
	double getStepError() { return 0.0; }

private:
    double* mTempStore1;

};

class EulerSolverInfo: public SolverInfo{
public:
	EulerSolverInfo() {mIsFSAL = 0; mVariableStep = 0; mStageCount = 1;}
};


#endif /* SOLVER_H_ */
