/*
 * Block.cpp
 *
 *  Created on: 17 янв. 2015 г.
 *      Author: frolov
 */

#include "Block.h"

Block::Block(int _length, int _width,
		int* _topBoundaryType, int* _leftBoundaryType, int* _bottomBoundaryType, int* _rightBoundaryType,
		double* _topBlockBoundary, double* _leftBlockBoundary, double* _bottomBlockBoundary, double* _rightBlockBoundary,
		double* _topExternalBoundary, double* _leftExternalBoundary, double* _bottomExternalBoundary, double* _rightExternalBoundary) {
	length = _length;
	width = _width;

	topBoundaryType = _topBoundaryType;
	leftBoundaryType = _leftBoundaryType;
	bottomBoundaryType = _bottomBoundaryType;
	rightBoundaryType = _rightBoundaryType;

	topBlockBoundary = _topBlockBoundary;
	leftBlockBoundary = _leftBlockBoundary;
	bottomBlockBoundary = _bottomBlockBoundary;
	rightBlockBoundary = _rightBlockBoundary;

	topExternalBoundary = _topExternalBoundary;
	leftExternalBoundary = _leftExternalBoundary;
	bottomExternalBoundary = _bottomExternalBoundary;
	rightExternalBoundary = _rightExternalBoundary;

	matrix = NULL;
}

Block::~Block() {

}
