/*
 * Block.cpp
 *
 *  Created on: 17 янв. 2015 г.
 *      Author: frolov
 */

#include "Block.h"

Block::Block() {
	length = width = 0;

	world_rank = 0;

	topBorderType = leftBorderType = bottomBorderType = rightBorderType = NULL;
	topBlockBorder = leftBlockBorder = bottomBlockBorder = rightBlockBorder = NULL;
	topExternalBorder = leftExternalBorder = bottomExternalBorder = rightExternalBorder = NULL;

	matrix = NULL;
}

Block::Block(int _world_rank) {
	world_rank = _world_rank;

	length = width = 0;

	topBorderType = leftBorderType = bottomBorderType = rightBorderType = NULL;
	topBlockBorder = leftBlockBorder = bottomBlockBorder = rightBlockBorder = NULL;
	topExternalBorder = leftExternalBorder = bottomExternalBorder = rightExternalBorder = NULL;

	matrix = NULL;
}

Block::Block(int _length, int _width, int _world_rank) {
	length = _length;
	width = _width;

	world_rank = _world_rank;

	topBorderType = leftBorderType = bottomBorderType = rightBorderType = NULL;
	topBlockBorder = leftBlockBorder = bottomBlockBorder = rightBlockBorder = NULL;
	topExternalBorder = leftExternalBorder = bottomExternalBorder = rightExternalBorder = NULL;

	matrix = NULL;
}

Block::~Block() {

}
