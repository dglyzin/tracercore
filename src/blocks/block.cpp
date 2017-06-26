/*
 * block.cpp
 *
 *  Created on: 12 окт. 2015 г.
 *      Author: frolov
 */

#include "block.h"

Block::Block(int _nodeNumber, int _dimension, int _xCount, int _yCount, int _zCount, int _cellSize, int _haloSize) {
	nodeNumber = _nodeNumber;

	xCount = _xCount;
	yCount = _yCount;
	zCount = _zCount;

	cellSize = _cellSize;
	haloSize = _haloSize;

	setCountAndOffset(_dimension);
}

Block::~Block() {
}

unsigned long long Block::getGridNodeCount() {
	return xCount * yCount * zCount;
}

unsigned long long Block::getGridElementCount() {
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

			break;

		case 2:
			zCount = 1;

			break;

		case 3:
			break;

		default:
			break;
	}
}
