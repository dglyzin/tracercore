/*
 * block.h
 *
 *  Created on: 12 окт. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKS_BLOCK_H_
#define SRC_BLOCKS_BLOCK_H_

#include "../problem/ordinary.h"

#include "../processingunit/processingunit.h"

class Block {
public:
	Block(int _dimension, int _xCount, int _yCount, int _zCount, int _xOffset, int _yOffset, int _zOffset, int _cellSize, int _haloSize);
	virtual ~Block();

	void computeStageBorder(int stage, double time);
	void computeStageCenter(int stage, double time);

	void prepareArgument(int stage, double timestep );

	void prepareStageData(int stage);

protected:
	int dimension;

	int xCount;
	int yCount;
	int zCount;

	int xOffset;
	int yOffset;
	int zOffset;

	int cellSize;
	int haloSize;
};

#endif /* SRC_BLOCKS_BLOCK_H_ */
