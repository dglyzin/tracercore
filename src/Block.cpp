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
