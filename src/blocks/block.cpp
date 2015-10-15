/*
 * block.cpp
 *
 *  Created on: 12 окт. 2015 г.
 *      Author: frolov
 */

#include "block.h"

Block::Block(int _dimension, int _xCount, int _yCount, int _zCount, int _xOffset, int _yOffset, int _zOffset, int _cellSize, int _haloSize) {
	dimension = _dimension;

	xCount = _xCount;
	yCount = _yCount;
	zCount = _zCount;

	xOffset = _xOffset;
	yOffset = _yOffset;
	zOffset = _zOffset;

	cellSize = _cellSize;
	haloSize = _haloSize;
}

Block::~Block() {
	// TODO Auto-generated destructor stub
}

int Block::getGridNodeCount() {
	return xCount * yCount * zCount;
}

int Block::getGridElementCount() {
	return getGridNodeCount() * cellSize;
}
