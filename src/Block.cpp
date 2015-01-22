/*
 * Block.cpp
 *
 *  Created on: 17 янв. 2015 г.
 *      Author: frolov
 */

#include "Block.h"

Block::Block(int _length, int _width) {
	length = _length;
	width = _width;

	/*topBorderType = _topBorderType;
	leftBorderType = _leftBorderType;
	bottomBorderType = _bottomBorderType;
	rightBorderType = _rightBorderType;*/

	/*topBlockBorder = _topBlockBorder;
	leftBlockBorder = _leftBlockBorder;
	bottomBlockBorder = _bottomBlockBorder;
	rightBlockBorder = _rightBlockBorder;

	topExternalBorder = _topExternalBorder;
	leftExternalBorder = _leftExternalBorder;
	bottomExternalBorder = _bottomExternalBorder;
	rightExternalBorder = _rightExternalBorder;*/

	topBorderType = leftBorderType = bottomBorderType = rightBorderType = NULL;
	topBlockBorder = leftBlockBorder = bottomBlockBorder = rightBlockBorder = NULL;
	topExternalBorder = leftExternalBorder = bottomExternalBorder = rightExternalBorder = NULL;

	matrix = NULL;
}

Block::~Block() {

}
