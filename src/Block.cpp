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

	borderType = NULL;
	blockBorder = NULL;
	externalBorder = NULL;

	matrix = NULL;
}

Block::Block(int _length, int _width, int _lengthMove, int _widthMove, int _nodeNumber) {
	length = _length;
	width = _width;

	lenghtMove = _lengthMove;
	widthMove = _widthMove;

	nodeNumber = _nodeNumber;

	borderType = NULL;
	blockBorder = NULL;
	externalBorder = NULL;

	matrix = NULL;
}

Block::~Block() {

}

void Block::prepareData() {
	if(!isRealBlock()) return;

	for (int i = 0; i < width; ++i)
		blockBorder[TOP][i] = matrix[0][i];

	for (int i = 0; i < length; ++i)
		blockBorder[LEFT][i] = matrix[i][0];

	for (int i = 0; i < width; ++i)
		blockBorder[BOTTOM][i] = matrix[length-1][i];

	for (int i = 0; i < length; ++i)
		blockBorder[RIGHT][i] = matrix[i][width-1];
}

void Block::setPartBorder(int type, int side, int move, int borderLength) {
	if( checkValue(side, move + borderLength) ) {
		printf("\nCritical error!\n");
		exit(1);
	}

	for (int i = 0; i < borderLength; ++i)
		borderType[side][i + move] = type;
}

double* Block::getBorderBlockData(int side, int move) {
	if( checkValue(side, move) ) {
		printf("\nCritical error!\n");
		exit(1);
	}

	return blockBorder != NULL ? blockBorder[side] + move : NULL;
}

double* Block::getExternalBorderData(int side, int move) {
	if( checkValue(side, move) ) {
		printf("\nCritical error!\n");
		exit(1);
	}

	return externalBorder != NULL ? externalBorder[side] + move : NULL;
}

bool Block::checkValue(int side, int move) {
	if( (side == TOP || side == BOTTOM) && move > width )
		return true;

	if( (side == LEFT || side == RIGHT) && move > length )
		return true;

	if( side >= BORDER_COUNT )
		return true;

	return false;
}
