/*
 * blockcpu3d.cpp
 *
 *  Created on: 28 авг. 2015 г.
 *      Author: frolov
 */

#include "blockcpu3d.h"

BlockCpu3d::BlockCpu3d(int _blockNumber, int _dimension, int _xCount, int _yCount, int _zCount,
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

BlockCpu3d::~BlockCpu3d() {
}

void BlockCpu3d::computeStageBorder(int stage, double time) {
# pragma omp parallel
	{
		double* result = mSolver->getStageResult(stage);
		double* source = mSolver->getStageSource(stage);

		for (int z = 0; z < haloSize; ++z) {
			int zShift = yCount * xCount * z;
	# pragma omp for
			for (int y = 0; y < yCount; ++y) {
				int yShift = xCount * y;
				for (int x = 0; x < xCount; ++x) {
					int xShift = x;
					//cout << "Border Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[ mCompFuncNumber[ zShift + yShift + xShift ] ](result, source, time, x, y, z, mParams, externalBorder);
				}
			}
		}

		for (int z = zCount - haloSize; z < zCount; ++z) {
			int zShift = yCount * xCount * z;
	# pragma omp for
			for (int y = 0; y < yCount; ++y) {
				int yShift = xCount * y;
				for (int x = 0; x < xCount; ++x) {
					int xShift = x;
					//cout << "Border Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[ mCompFuncNumber[ zShift + yShift + xShift ] ](result, source, time, x, y, z, mParams, externalBorder);
				}
			}
		}

# pragma omp for
		for (int z = haloSize; z < zCount - haloSize; ++z) {
			int zShift = yCount * xCount * z;
			for (int y = 0; y < haloSize; ++y) {
				int yShift = xCount * y;
				for (int x = 0; x < xCount; ++x) {
					int xShift = x;
					//cout << "Border Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[ mCompFuncNumber[ zShift + yShift + xShift ] ](result, source, time, x, y, z, mParams, externalBorder);
				}
			}
		}

# pragma omp for
		for (int z = haloSize; z < zCount - haloSize; ++z) {
			int zShift = yCount * xCount * z;
			for (int y = yCount - haloSize; y < yCount; ++y) {
				int yShift = xCount * y;
				for (int x = 0; x < xCount; ++x) {
					int xShift = x;
					//cout << "Border Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[ mCompFuncNumber[ zShift + yShift + xShift ] ](result, source, time, x, y, z, mParams, externalBorder);
				}
			}
		}

# pragma omp for
		for (int z = haloSize; z < zCount - haloSize; ++z) {
			int zShift = yCount * xCount * z;
			for (int y = haloSize; y < yCount - haloSize; ++y) {
				int yShift = xCount * y;
				for (int x = 0; x < haloSize; ++x) {
					int xShift = x;
					//cout << "Border Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[ mCompFuncNumber[ zShift + yShift + xShift ] ](result, source, time, x, y, z, mParams, externalBorder);
				}
			}
		}

# pragma omp for
		for (int z = haloSize; z < zCount - haloSize; ++z) {
			int zShift = yCount * xCount * z;
			for (int y = haloSize; y < yCount - haloSize; ++y) {
				int yShift = xCount * y;
				for (int x = xCount - haloSize; x < xCount; ++x) {
					int xShift = x;
					//cout << "Border Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[ mCompFuncNumber[ zShift + yShift + xShift ] ](result, source, time, x, y, z, mParams, externalBorder);
				}
			}
		}
	}
}

void BlockCpu3d::computeStageCenter(int stage, double time) {
# pragma omp parallel
	{
		double* result = mSolver->getStageResult(stage);
		double* source = mSolver->getStageSource(stage);
# pragma omp for
		for (int z = haloSize; z < zCount - haloSize; ++z) {
			int zShift = yCount * xCount * z;
			for (int y = haloSize; y < yCount - haloSize; ++y) {
				int yShift = xCount * y;
				for (int x = haloSize; x < xCount - haloSize; ++x) {
					int xShift = x;
					//cout << "Calc z_" << z << " y_" << y << " x_" << x << endl;
					mUserFuncs[ mCompFuncNumber[ zShift + yShift + xShift ] ](result, source, time, x, y, z, mParams, externalBorder);
				}
			}
		}
	}
}
