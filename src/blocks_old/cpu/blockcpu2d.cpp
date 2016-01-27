/*
 * blockcpu2d.cpp
 *
 *  Created on: 28 авг. 2015 г.
 *      Author: frolov
 */

#include "../../blocks_old/cpu/blockcpu2d.h"

BlockCpu2d::BlockCpu2d(int _blockNumber, int _dimension, int _xCount, int _yCount, int _zCount,
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

BlockCpu2d::~BlockCpu2d() {
}

void BlockCpu2d::computeStageBorder(int stage, double time) {
# pragma omp parallel
	{
		double* result = mSolver->getStageResult(stage);
		double* source0 = mSolver->getStageSource(stage);
# pragma omp for
		for (int x = 0; x < xCount; ++x) {
			int xShift = x;
			for (int y = 0; y < haloSize; ++y) {
				int yShift = xCount * y;
				//cout << "Calc y_" << y << " x_" << x << endl;
				//printf("Calc y = %d, x = %d\n", y, x);
				mUserFuncs[ mCompFuncNumber[ yShift + xShift ] ](result, &source0, time, x, y, 0, mParams, externalBorder);
			}
		}

# pragma omp for
		for (int x = 0; x < xCount; ++x) {
			int xShift = x;
			for (int y = yCount - haloSize; y < yCount; ++y) {
				int yShift = xCount * y;
				//cout << "Calc y_" << y << " x_" << x << endl;
				//printf("Calc y = %d, x = %d\n", y, x);
				mUserFuncs[ mCompFuncNumber[ yShift + xShift ] ](result, &source0, time, x, y, 0, mParams, externalBorder);
			}
		}

# pragma omp for
		for (int y = haloSize; y < yCount - haloSize; ++y) {
			int yShift = xCount * y;
			for (int x = 0; x < haloSize; ++x) {
				int xShift = x;
				//cout << "Calc y_" << y << " x_" << x << endl;
				//printf("Calc y = %d, x = %d\n", y, x);
				mUserFuncs[ mCompFuncNumber[ yShift + xShift ] ](result, &source0, time, x, y, 0, mParams, externalBorder);
			}
		}

# pragma omp for
		for (int y = haloSize; y < yCount - haloSize; ++y) {
			int yShift = xCount * y;
			for (int x = xCount - haloSize; x < xCount; ++x) {
				int xShift = x;
				//cout << "Calc y_" << y << " x_" << x << endl;
				//printf("Calc y = %d, x = %d\n", y, x);
				mUserFuncs[ mCompFuncNumber[ yShift + xShift ] ](result, &source0, time, x, y, 0, mParams, externalBorder);
			}
		}
	}
}

void BlockCpu2d::computeStageCenter(int stage, double time) {
# pragma omp parallel
	{
		double* result = mSolver->getStageResult(stage);
		double* source0 = mSolver->getStageSource(stage);
# pragma omp for
		for (int y = haloSize; y < yCount - haloSize; ++y) {
			int yShift = xCount * y;
			for (int x = haloSize; x < xCount - haloSize; ++x) {
				int xShift = x;
				//cout << "Calc y_" << y << " x_" << x << endl;
				mUserFuncs[ mCompFuncNumber[ yShift + xShift ] ](result, &source0, time, x, y, 0, mParams, externalBorder);
			}
		}
	}
}
