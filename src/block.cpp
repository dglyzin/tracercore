/*
 * Block.cpp
 *
 *  Created on: 17 янв. 2015 г.
 *      Author: frolov
 */

#include "block.h"
#include <cuda.h>
#include <cuda_runtime_api.h>


Block::Block(int _dimension, int _xCount, int _yCount, int _zCount, int _xOffset, int _yOffset, int _zOffset, int _nodeNumber, int _deviceNumber) {
	dimension = _dimension;

	xCount = _xCount;
	yCount = _yCount;
	zCount = _zCount;

	xOffset = _xOffset;
	yOffset = _yOffset;
	zOffset = _zOffset;

	nodeNumber = _nodeNumber;

	deviceNumber = _deviceNumber;

	countSendSegmentBorder = countReceiveSegmentBorder = 0;

	sendBorderType = NULL;
	receiveBorderType = NULL;

	blockBorder = NULL;
	externalBorder = NULL;

	blockBorderMove = NULL;
	externalBorderMove = NULL;
	
	blockBorderMemoryAllocType = NULL;
	externalBorderMemoryAllocType = NULL;

	matrix = newMatrix = NULL;
	functionNumber = NULL;
	
	//result = NULL;
}

Block::~Block() {

}

int Block::getGridNodeCount() {
	switch (dimension) {
		case 0:
			return xCount;
		case 1:
			return xCount * yCount;
		case 2:
			return xCount * yCount * zCount;
		default:
			return xCount;
	}
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
