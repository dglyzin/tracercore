/*
 * blockcpu2d.cpp
 *
 *  Created on: 28 авг. 2015 г.
 *      Author: frolov
 */

#include "blockcpu2d.h"

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

