/*
 * Block.cpp
 *
 *  Created on: 17 янв. 2015 г.
 *      Author: frolov
 */

#include "block.h"
#include <cuda.h>
#include <cuda_runtime_api.h>


Block::Block(int _dimension, int _xCount, int _yCount, int _zCount,
		int _xOffset, int _yOffset, int _zOffset,
		int _nodeNumber, int _deviceNumber,
		int _haloSize, int _cellSize) {
	dimension = _dimension;

	switch (dimension) {
		case 1:
			xCount = _xCount;
			yCount = 1;
			zCount = 1;

			xOffset = _xOffset;
			yOffset = 0;
			zOffset = 0;

			break;

		case 2:
			xCount = _xCount;
			yCount = _yCount;
			zCount = 1;

			xOffset = _xOffset;
			yOffset = _yOffset;
			zOffset = 0;

			break;

		case 3:
			xCount = _xCount;
			yCount = _yCount;
			zCount = _zCount;

			xOffset = _xOffset;
			yOffset = _yOffset;
			zOffset = _zOffset;

			break;
		default:
			break;
	}

	nodeNumber = _nodeNumber;

	deviceNumber = _deviceNumber;

	countSendSegmentBorder = countReceiveSegmentBorder = 0;

	sendBorderInfo = NULL;
	receiveBorderInfo = NULL;

	blockBorder = NULL;
	externalBorder = NULL;
	
	blockBorderMemoryAllocType = NULL;
	externalBorderMemoryAllocType = NULL;

	mCompFuncNumber = NULL;

	cellSize = _cellSize;
	haloSize = _haloSize;

	mParamsCount = 0;
	mParams = NULL;

	mUserFuncs = NULL;
	mUserInitFuncs = NULL;
}

Block::~Block() {

}

int Block::getGridNodeCount() {
	return xCount * yCount * zCount;
}

int Block::getGridElementCount() {
	return getGridNodeCount() * cellSize;
}

void Block::confirmStep() {
	mSolver->confirmStep();
}

void Block::freeMemory(int memory_alloc_type, double* memory) {
	if(memory == NULL)
		return;
	
	switch(memory_alloc_type) {
		case NOT_ALLOC:
			break;
			
		case NEW:
			delete memory;
			break;
			
		case CUDA_MALLOC:
			cudaFree(memory);
			break;
			
		case CUDA_MALLOC_HOST:
			cudaFreeHost(memory);
			break;
			
		default:
			break;
	}
}

void Block::freeMemory(int memory_alloc_type, int* memory) {
	if(memory == NULL)
		return;
	
	switch(memory_alloc_type) {
		case NOT_ALLOC:
			break;
			
		case NEW:
			delete memory;
			break;
			
		case CUDA_MALLOC:
			cudaFree(memory);
			break;
			
		case CUDA_MALLOC_HOST:
			cudaFreeHost(memory);
			break;
			
		default:
			break;
	}
}
