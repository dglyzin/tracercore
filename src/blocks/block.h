/*
 * block.h
 *
 *  Created on: 12 окт. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKS_BLOCK_H_
#define SRC_BLOCKS_BLOCK_H_

#include <stdlib.h>
#include <stdio.h>
#include <fstream>
#include "../processingunit/processingunit.h"

class Block {
public:
	Block(int _nodeNumber, int _dimension, int _xCount, int _yCount, int _zCount, int _xOffset, int _yOffset,
			int _zOffset, int _cellSize, int _haloSize);
	virtual ~Block();

	//virtual void afterCreate(int problemType, int solverType, double aTol, double rTol) = 0;

	virtual void computeStageBorder(int stage, double time) = 0;
	virtual void computeStageCenter(int stage, double time) = 0;

	virtual void prepareArgument(int stage, double timestep) = 0;
	virtual void getSubVolume(double* result, int zStart, int zStop, int yStart, int yStop, int xStart,
				         int xStop, int yCount, int xCount, int cellSize) = 0;
	virtual void setSubVolume(double* source, int zStart, int zStop, int yStart, int yStop, int xStart,
				int xStop, int yCount, int xCount, int cellSize) = 0;


	virtual void prepareStageData(int stage) = 0;
	virtual void prepareStageSourceResult(int stage, double timeStep, double currentTime) = 0;

	virtual bool isRealBlock() = 0;
	virtual int getBlockType() = 0;
	virtual int getDeviceNumber() = 0;

	virtual bool isBlockType(int type) = 0;
	virtual bool isDeviceNumber(int number) = 0;

	virtual bool isProcessingUnitCPU() = 0;
	virtual bool isProcessingUnitGPU() = 0;

	virtual double getStepError(double timestep) = 0;

	virtual void confirmStep(double timestep) = 0;
	virtual void rejectStep(double timestep) = 0;

	virtual double* addNewBlockBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength,
			int nLength) = 0;
	virtual double* addNewExternalBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength,
			double* border) = 0;

	virtual void moveTempBorderVectorToBorderArray() = 0;

	virtual void getCurrentState(double* result) = 0;

	virtual void saveStateForDraw(char* path) = 0;
	virtual void saveStateForLoad(char* path) = 0;
	virtual void saveStateForDrawDenseOutput(char* path, double timestep, double tetha) = 0;
	virtual void loadState(std::ifstream& in) = 0;

	virtual bool isNan() = 0;

	int getGridNodeCount();
	int getGridElementCount();

	int getNodeNumber();

	virtual ProcessingUnit* getPU() = 0;

	virtual void print() = 0;

protected:
	int nodeNumber;

	int xCount;
	int yCount;
	int zCount;

	int xOffset;
	int yOffset;
	int zOffset;

	int cellSize;
	int haloSize;

private:
	void setCountAndOffset(int dimension);
};

#endif /* SRC_BLOCKS_BLOCK_H_ */
