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
    virtual double getStageTimeStep(int stage) { return 0; }
    virtual void prepareArgument(int stage, double timeStep) { return; }
    virtual void confirmStep(double timestep) { return; }
    virtual double getStepError(double timeStep, double aTol, double rTol) { return 0.0; }


protected:
  	int     mCount; //total number of elements in every array
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

	virtual double getNewStep(double timeStep, double error, int totalDomainElements) { return timeStep; }
	virtual int isErrorOK(double error, int totalDomainElements) { return 1; }

protected:
	int mIsFSAL;
	int mVariableStep;
	int mStageCount;
};

/*Solver* GetCpuSolver(int solverIdx, int count);
Solver* GetGpuSolver(int solverIdx, int count);*/

//***********************1. EULER SOLVER**************
class EulerSolver: public Solver{
public:
	EulerSolver(int _count);
	~EulerSolver();
	double* getStageSource(int stage);
	double* getStageResult(int stage);
	void prepareArgument(int stage, double timeStep);
	void confirmStep(double timestep);
	double getStepError(double timeStep, double aTol, double rTol) { return 0.0; }

private:
    double* mTempStore1;

};

class EulerSolverInfo: public SolverInfo{
public:
	EulerSolverInfo() {mIsFSAL = 0; mVariableStep = 0; mStageCount = 1;}
};



//***********************2. RK4 SOLVER**************
class RK4Solver: public Solver{
public:
	RK4Solver(int _count);
	~RK4Solver();
	double* getStageSource(int stage);
	double* getStageResult(int stage);
	double getStageTimeStep(int stage);
	void prepareArgument(int stage, double timeStep);
	void confirmStep(double timestep);
	double getStepError(double timeStep, double aTol, double rTol) { return 0.0; }

private:
    double* mTempStore1;
    double* mTempStore2;
    double* mTempStore3;
    double* mTempStore4;
    double* mArg;

};

class RK4SolverInfo: public SolverInfo{
public:
	RK4SolverInfo() {mIsFSAL = 0; mVariableStep = 0; mStageCount = 4;}
};


//***********************3. DP45 SOLVER**************
class DP45Solver: public Solver{
public:
	DP45Solver(int _count);
	~DP45Solver();
	double* getStageSource(int stage);
	double* getStageResult(int stage);
	double getStageTimeStep(int stage);
	void prepareArgument(int stage, double timeStep);
	void confirmStep(double timestep);
	double getStepError(double timeStep, double aTol, double rTol);

private:
    double* mTempStore1;
    double* mTempStore2;
    double* mTempStore3;
    double* mTempStore4;
    double* mTempStore5;
    double* mTempStore6;
    double* mTempStore7;
    double* mArg;

};

class DP45SolverInfo: public SolverInfo{
public:
	DP45SolverInfo() {mIsFSAL = 1; mVariableStep = 1; mStageCount = 6;}
	double getNewStep(double timeStep, double error, int totalDomainElements); //error = total sum of squares from all blocks
	int isErrorOK(double error, int totalDomainElements); //error = total sum of squares from all blocks

};


#endif /* SOLVER_H_ */
