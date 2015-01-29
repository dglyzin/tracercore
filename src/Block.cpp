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

void Block::print() {
	if(!isRealBlock()) return;

	printf("FROM NODE #%d", nodeNumber);

	printf("\nLength: %d, Width: %d, World_Rank: %d\n", length, width, nodeNumber);

	printf("\nMatrix:\n");
	for (int i = 0; i < length; ++i)
	{
		for (int j = 0; j < width; ++j)
			printf("%6.1f ", matrix[i][j]);
		printf("\n");
	}


	printf("\ntopBorderType\n");
	for (int i = 0; i < width; ++i)
		printf("%4d", borderType[TOP][i]);
	printf("\n");


	printf("\nleftBorderType\n");
	for (int i = 0; i < length; ++i)
		printf("%4d", borderType[LEFT][i]);
	printf("\n");


	printf("\nbottomBorderType\n");
	for (int i = 0; i < width; ++i)
		printf("%4d", borderType[BOTTOM][i]);
	printf("\n");


	printf("\nrightBorderType\n");
	for (int i = 0; i < length; ++i)
		printf("%4d", borderType[RIGHT][i]);
	printf("\n");


	printf("\ntopBlockBorder\n");
	for (int i = 0; i < width; ++i)
		printf("%6.1f", blockBorder[TOP][i]);
	printf("\n");


	printf("\nleftBlockBorder\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", blockBorder[LEFT][i]);
	printf("\n");


	printf("\nbottomBlockBorder\n");
	for (int i = 0; i <width; ++i)
		printf("%6.1f", blockBorder[BOTTOM][i]);
	printf("\n");


	printf("\nrightBlockBorder\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", blockBorder[RIGHT][i]);
	printf("\n");


	printf("\ntopExternalBorder\n");
	for (int i = 0; i < width; ++i)
		printf("%6.1f", externalBorder[TOP][i]);
	printf("\n");


	printf("\nleftExternalBorder\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", externalBorder[LEFT][i]);
	printf("\n");


	printf("\nbottomExternalBorder\n");
	for (int i = 0; i <width; ++i)
		printf("%6.1f", externalBorder[BOTTOM][i]);
	printf("\n");


	printf("\nrightExternalBorder\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", externalBorder[RIGHT][i]);
	printf("\n");


	printf("\n\n\n\n\n\n\n");
}

void Block::printMatrix() {
	if(!isRealBlock()) return;

	printf("\nMatrix:\n");
	for (int i = 0; i < length; ++i)
	{
		for (int j = 0; j < width; ++j)
			printf("%6.1f ", matrix[i][j]);
		printf("\n");
	}
}
