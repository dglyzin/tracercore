/*
 * nullblock.h
 *
 *  Created on: 05 нояб. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKS_NULLBLOCK_H_
#define SRC_BLOCKS_NULLBLOCK_H_

#include <stdlib.h>
#include <stdio.h>

#include "../enums.h"

#include "block.h"

class NullBlock: public Block {
public:
	NullBlock(int _nodeNumber, int _dimension, int _xCount, int _yCount, int _zCount, int _xOffset, int _yOffset, int _zOffset, int _cellSize, int _haloSize);
	virtual ~NullBlock();

	void computeStageBorder(int stage, double time);
	void computeStageCenter(int stage, double time);

	void prepareArgument(int stage, double timestep );

	void prepareStageData(int stage);


	bool isRealBlock();
	int getBlockType();
	int getDeviceNumber();

	bool isProcessingUnitCPU();
	bool isProcessingUnitGPU();

	double getStepError(double timestep);

	void confirmStep(double timestep);
	void rejectStep(double timestep);

	double* addNewBlockBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength);
	double* addNewExternalBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength, double* border);

	void moveTempBorderVectorToBorderArray();

	void loadData(double* data);
	void getCurrentState(double* result);

	void saveState(char* path);
	void loadState(std::ifstream& in);
};

#endif /* SRC_BLOCKS_NULLBLOCK_H_ */
