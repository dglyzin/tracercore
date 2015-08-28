/*
 * blockcpu3d.h
 *
 *  Created on: 28 авг. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKS_BLOCKCPU3D_H_
#define SRC_BLOCKS_BLOCKCPU3D_H_

#include "blockcpu.h"

class BlockCpu3d: public BlockCpu {
public:
	BlockCpu3d(int _blockNumber, int _dimension, int _xCount, int _yCount, int _zCount,
			int _xOffset, int _yOffset, int _zOffset,
			int _nodeNumber, int _deviceNumber,
			int _haloSize, int _cellSize,
			unsigned short int* _initFuncNumber, unsigned short int* _compFuncNumber,
			int _solverIdx, double _aTol, double _rTol);

	virtual ~BlockCpu3d();
};

#endif /* SRC_BLOCKS_BLOCKCPU3D_H_ */
