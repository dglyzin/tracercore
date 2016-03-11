/*
 * block.cpp
 *
 *  Created on: 12 окт. 2015 г.
 *      Author: frolov
 */

#include "block.h"

Block::Block(int _nodeNumber, int _dimension, int _xCount, int _yCount, int _zCount, int _xOffset, int _yOffset,
		int _zOffset, int _cellSize, int _haloSize) {
	nodeNumber = _nodeNumber;

	xCount = _xCount;
	yCount = _yCount;
	zCount = _zCount;

	xOffset = _xOffset;
	yOffset = _yOffset;
	zOffset = _zOffset;

	cellSize = _cellSize;
	haloSize = _haloSize;

	setCountAndOffset(_dimension);
}

Block::~Block() {
}

int Block::getGridNodeCount() {
	return xCount * yCount * zCount;
}

int Block::getGridElementCount() {
	return getGridNodeCount() * cellSize;
}

int Block::getNodeNumber() {
	return nodeNumber;
}

void Block::setCountAndOffset(int dimension) {
	switch (dimension) {
		case 1:
			yCount = 1;
			zCount = 1;

			yOffset = 0;
			zOffset = 0;

			break;

		case 2:
			zCount = 1;

			zOffset = 0;

			break;

		case 3:
			break;

		default:
			break;
	}
}
