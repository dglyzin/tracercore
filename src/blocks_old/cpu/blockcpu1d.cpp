/*
 * blockcpu1d.cpp
 *
 *  Created on: 28 авг. 2015 г.
 *      Author: frolov
 */

#include "../../blocks_old/cpu/blockcpu1d.h"

BlockCpu1d::BlockCpu1d(int _blockNumber, int _dimension, int _xCount, int _yCount, int _zCount,
		int _xOffset, int _yOffset, int _zOffset,
		int _nodeNumber, int _deviceNumber,
		int _haloSize, int _cellSize,
		unsigned short int* _initFuncNumber, unsigned short int* _compFuncNumber,
		int _solverIdx, double _aTol, double _rTol) :
				BlockCpu( _blockNumber, _dimension, _xCount, _yCount, _zCount,
				_xOffset, _yOffset, _zOffset,
				_nodeNumber, _deviceNumber,
				_haloSize, _cellSize,
				_initFuncNumber, _compFuncNumber,
				_solverIdx, _aTol, _rTol) {
}

BlockCpu1d::~BlockCpu1d() {
}

void BlockCpu1d::computeStageBorder(int stage, double time) {
	//char c;
	//cout << endl << endl << "before error?" << blockNumber << endl;
	//scanf("%c", &c);
# pragma omp parallel
	{
		double* result = mSolver->getStageResult(stage);
		double* source0 = mSolver->getStageSource(stage);
# pragma omp for
		for (int x = 0; x < haloSize; ++x) {
			//cout << "Border Calc x_" << x << endl;
			mUserFuncs[ mCompFuncNumber[x] ](result, &source0, time, x, 0, 0, mParams, externalBorder);
		}

# pragma omp for
		for (int x = xCount - haloSize; x < xCount; ++x) {
			//cout << "Border Calc x_" << x << endl;
			mUserFuncs[ mCompFuncNumber[x] ](result, &source0, time, x, 0, 0, mParams, externalBorder);
		}
	}

	//scanf("%c", &c);
}

void BlockCpu1d::computeStageCenter(int stage, double time) {
# pragma omp parallel
	{
		double* result = mSolver->getStageResult(stage);
		double* source0 = mSolver->getStageSource(stage);
# pragma omp for
		for (int x = haloSize; x < xCount - haloSize; ++x) {
			//cout << "Calc x_" << x << endl;
			mUserFuncs[ mCompFuncNumber[x] ](result, &source0, time, x, 0, 0, mParams, externalBorder);
		}
	}
}
