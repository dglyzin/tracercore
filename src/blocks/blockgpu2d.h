/*
 * blockgpu2d.h
 *
 *  Created on: 29 авг. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKS_BLOCKGPU2D_H_
#define SRC_BLOCKS_BLOCKGPU2D_H_

#include "blockgpu.h"

class BlockGpu2d: public BlockGpu {
public:
	BlockGpu2d(int _blockNumber, int _dimension, int _xCount, int _yCount, int _zCount,
			int _xOffset, int _yOffset, int _zOffset,
			int _nodeNumber, int _deviceNumber,
			int _haloSize, int _cellSize,
			unsigned short int* _initFuncNumber, unsigned short int* _compFuncNumber,
			int _solverIdx, double _aTol, double _rTol);

	virtual ~BlockGpu2d();
};

#endif /* SRC_BLOCKS_BLOCKGPU2D_H_ */
