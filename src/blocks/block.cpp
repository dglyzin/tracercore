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

	//dimension = _dimension;

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
	// TODO Auto-generated destructor stub
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

void Block::printToConsole() {
	printf("\nBlock\n"
			"Node number: %d\n"
			"xCount:      %d\n"
			"yCount:      %d\n"
			"zCount:      %d\n"
			"xOffset:     %d\n"
			"yOffset:     %d\n"
			"zOffset:     %d\n"
			"Cell size:   %d\n"
			"Halo size:   %d\n"
			"\n", nodeNumber, xCount, yCount, zCount, xOffset, yOffset, zOffset, cellSize, haloSize);
}

void Block::setCountAndOffset(int dimension) {
	switch (dimension) {
		case 1:
			//xCount = _xCount;
			yCount = 1;
			zCount = 1;

			//xOffset = _xOffset;
			yOffset = 0;
			zOffset = 0;

			break;

		case 2:
			//xCount = _xCount;
			//yCount = _yCount;
			zCount = 1;

			//xOffset = _xOffset;
			//yOffset = _yOffset;
			zOffset = 0;

			break;

		case 3:
			//xCount = _xCount;
			//yCount = _yCount;
			//zCount = _zCount;

			//xOffset = _xOffset;
			//yOffset = _yOffset;
			//zOffset = _zOffset;

			break;

		default:
			break;
	}
}
