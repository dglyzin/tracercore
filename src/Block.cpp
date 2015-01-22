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

	/*topBoundaryType = _topBoundaryType;
	leftBoundaryType = _leftBoundaryType;
	bottomBoundaryType = _bottomBoundaryType;
	rightBoundaryType = _rightBoundaryType;*/

	/*topBlockBoundary = _topBlockBoundary;
	leftBlockBoundary = _leftBlockBoundary;
	bottomBlockBoundary = _bottomBlockBoundary;
	rightBlockBoundary = _rightBlockBoundary;

	topExternalBoundary = _topExternalBoundary;
	leftExternalBoundary = _leftExternalBoundary;
	bottomExternalBoundary = _bottomExternalBoundary;
	rightExternalBoundary = _rightExternalBoundary;*/

	topBoundaryType = leftBoundaryType = bottomBoundaryType = rightBoundaryType = NULL;
	topBlockBoundary = leftBlockBoundary = bottomBlockBoundary = rightBlockBoundary = NULL;
	topExternalBoundary = leftExternalBoundary = bottomExternalBoundary = rightExternalBoundary = NULL;

	matrix = NULL;
}

Block::~Block() {

}
