/*
 * Block.cpp
 *
 *  Created on: 17 янв. 2015 г.
 *      Author: frolov
 */

#include "Block.h"

Block::Block() {
	length = width = 0;

	nodeNumber = 0;

	lenghtMove = widthMove = 0;

	topBorderType = leftBorderType = bottomBorderType = rightBorderType = NULL;
	topBlockBorder = leftBlockBorder = bottomBlockBorder = rightBlockBorder = NULL;
	topExternalBorder = leftExternalBorder = bottomExternalBorder = rightExternalBorder = NULL;

	matrix = NULL;
}

Block::Block(int _length, int _width, int _lengthMove, int _widthMove, int _nodeNumber) {
	length = _length;
	width = _width;

	lenghtMove = _lengthMove;
	widthMove = _widthMove;

	nodeNumber = _nodeNumber;

	topBorderType = leftBorderType = bottomBorderType = rightBorderType = NULL;
	topBlockBorder = leftBlockBorder = bottomBlockBorder = rightBlockBorder = NULL;
	topExternalBorder = leftExternalBorder = bottomExternalBorder = rightExternalBorder = NULL;

	matrix = NULL;
}

Block::~Block() {

}
