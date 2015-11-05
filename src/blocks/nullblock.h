/*
 * nullblock.h
 *
 *  Created on: 05 нояб. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKS_NULLBLOCK_H_
#define SRC_BLOCKS_NULLBLOCK_H_

#include "block.h"

class NullBlock: public Block {
public:
	NullBlock(int _nodeNumber, int _dimension, int _xCount, int _yCount, int _zCount, int _xOffset, int _yOffset, int _zOffset, int _cellSize, int _haloSize);
	virtual ~NullBlock();
};

#endif /* SRC_BLOCKS_NULLBLOCK_H_ */
