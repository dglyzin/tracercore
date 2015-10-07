/*
 * dp45storage.h
 *
 *  Created on: 06 окт. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_STEPSTORAGE_DP45STORAGE_H_
#define SRC_STEPSTORAGE_DP45STORAGE_H_

#include "stepstorage.h"

class DP45Storage: public StepStorage {
public:
	DP45Storage();
	DP45Storage(ProcessingUnit* pu, int count, double _aTol, double _rTol);
	virtual ~DP45Storage();

    double* getStageSource(int stage);
    double* getStageResult(int stage);

    double getStageTimeStep(int stage);

    void prepareArgument(ProcessingUnit* pu, int stage, double timestep);

    void confirmStep(ProcessingUnit* pu, double timestep);
    void rejectStep(ProcessingUnit* pu, double timestep);

    double getStepError(ProcessingUnit* pu, double timestep);

    bool isFSAL();
    bool isVariableStep();
    int getStageCount();

	double getNewStep(double timestep, double error, int totalDomainElements);
	bool isErrorPermissible(double error, int totalDomainElements);

	double* getDenseOutput(StepStorage* secondState);

private:
    double* mTempStore1;
    double* mTempStore2;
    double* mTempStore3;
    double* mTempStore4;
    double* mTempStore5;
    double* mTempStore6;
    double* mTempStore7;
    double* mArg;


    static const double c2=0.2, c3=0.3, c4=0.8, c5=8.0/9.0;

    static const double a21=0.2, a31=3.0/40.0, a32=9.0/40.0;
    static const double a41=44.0/45.0, a42=-56.0/15.0, a43=32.0/9.0;
    static const double a51=19372.0/6561.0, a52=-25360.0/2187.0;
    static const double a53=64448.0/6561.0, a54=-212.0/729.0;
    static const double a61=9017.0/3168.0, a62=-355.0/33.0, a63=46732.0/5247.0;
    static const double a64=49.0/176.0, a65=-5103.0/18656.0;
    static const double a71=35.0/384.0, a73=500.0/1113.0, a74=125.0/192.0;
    static const double a75=-2187.0/6784.0, a76=11.0/84.0;
    static const double e1=71.0/57600.0, e3=-71.0/16695.0, e4=71.0/1920.0;
    static const double e5=-17253.0/339200.0, e6=22.0/525.0, e7=-1.0/40.0;
    static const double facmin=0.5, facmax = 2, fac = 0.9;

    void prepareFSAL(ProcessingUnit* pu, double timestep);
};

#endif /* SRC_STEPSTORAGE_DP45STORAGE_H_ */
