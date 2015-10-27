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
	Block(int _nodeNumber, int _dimension, int _xCount, int _yCount, int _zCount, int _xOffset, int _yOffset, int _zOffset, int _cellSize, int _haloSize);
	virtual ~Block();

	virtual void computeStageBorder(int stage, double time) = 0;
	virtual void computeStageCenter(int stage, double time) = 0;

	virtual void prepareArgument(int stage, double timestep ) = 0;

	virtual void prepareStageData(int stage) = 0;


	virtual bool isRealBlock() = 0;
	virtual int getBlockType() = 0;

	virtual bool isProcessingUnitCPU() = 0;
	virtual bool isProcessingUnitGPU() = 0;

	virtual double getStepError(double timestep) = 0;

	virtual void confirmStep(double timestep) = 0;
	virtual void rejectStep(double timestep) = 0;

	virtual double* addNewBlockBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength) = 0;
	virtual double* addNewExternalBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength, double* border) = 0;

	virtual void moveTempBorderVectorToBorderArray() = 0;

	virtual void loadData(double* data) = 0;
	virtual void getCurrentState(double* result) = 0;

	int getGridNodeCount();
	int getGridElementCount();

	int getNodeNumber();

protected:
	int nodeNumber;

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
