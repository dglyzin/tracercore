/*
 * Block.cpp
 *
 *  Created on: 17 янв. 2015 г.
 *      Author: frolov
 */

#include "block.h"

using namespace std;


Block::Block(int _blockNumber, int _dimension, int _xCount, int _yCount, int _zCount,
		int _xOffset, int _yOffset, int _zOffset,
		int _nodeNumber, int _deviceNumber,
		int _haloSize, int _cellSize) {
	blockNumber = _blockNumber;
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

	mSolver = NULL;
}

Block::~Block() {

}

int Block::getGridNodeCount() {
	return xCount * yCount * zCount;
}

int Block::getGridElementCount() {
	return getGridNodeCount() * cellSize;
}

void Block::prepareArgument(int stage, double timestep){
	mSolver->prepareArgument(stage, timestep);
}

void Block::confirmStep(double timestep) {
	mSolver->confirmStep( timestep);
}

void Block::rejectStep(double timestep) {
	mSolver->rejectStep( timestep);
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

void Block::computeStageCenter(int stage, double time) {
	switch (dimension) {
		case 1:
			computeStageCenter_1d(stage, time);
			break;
		case 2:
			computeStageCenter_2d(stage, time);
			break;
		case 3:
			computeStageCenter_3d(stage, time);
			break;
		default:
			break;
	}
}

void Block::computeStageBorder(int stage, double time) {
	switch (dimension) {
		case 1:
			computeStageBorder_1d(stage, time);
			break;
		case 2:
			computeStageBorder_2d(stage, time);
			break;
		case 3:
			computeStageBorder_3d(stage, time);
			break;
		default:
			break;
	}
}

double* Block::addNewBlockBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength) {
	countSendSegmentBorder++;

	tempSendBorderInfo.push_back(side);
	tempSendBorderInfo.push_back(mOffset);
	tempSendBorderInfo.push_back(nOffset);
	tempSendBorderInfo.push_back(mLength);
	tempSendBorderInfo.push_back(nLength);

	int borderLength = mLength * nLength * cellSize * haloSize;
	int memoryType = NOT_ALLOC;

	double* newBlockBorder = getNewBlockBorder(neighbor, borderLength, memoryType);

	/*if( ( nodeNumber == neighbor->getNodeNumber() ) && isGPU( neighbor->getBlockType() ) ) {
		cudaMallocHost ( (void**)&newBlockBorder, borderLength * sizeof(double) );
		tempBlockBorderMemoryAllocType.push_back(CUDA_MALLOC_HOST);
	}
	else {
		newBlockBorder = new double [borderLength];
		tempBlockBorderMemoryAllocType.push_back(NEW);
	}*/

	tempBlockBorder.push_back(newBlockBorder);
	tempBlockBorderMemoryAllocType.push_back(memoryType);

	return newBlockBorder;
}

double* Block::addNewExternalBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength, double* border) {
	countReceiveSegmentBorder++;

	tempReceiveBorderInfo.push_back(side);
	tempReceiveBorderInfo.push_back(mOffset);
	tempReceiveBorderInfo.push_back(nOffset);
	tempReceiveBorderInfo.push_back(mLength);
	tempReceiveBorderInfo.push_back(nLength);

	int borderLength = mLength * nLength * cellSize * haloSize;
	int memoryType = NOT_ALLOC;

	double* newExternalBorder = getNewExternalBorder(neighbor, borderLength, border, memoryType);

	/*if( nodeNumber == neighbor->getNodeNumber() ) {
		newExternalBorder = border;
		tempExternalBorderMemoryAllocType.push_back(NOT_ALLOC);
	}
	else {
		newExternalBorder = new double [borderLength];
		tempExternalBorderMemoryAllocType.push_back(NEW);
	}*/

	tempExternalBorder.push_back(newExternalBorder);
	tempExternalBorderMemoryAllocType.push_back(memoryType);

	return newExternalBorder;
}

void Block::prepareStageData(int stage) {
	for (int i = 0; i < countSendSegmentBorder; ++i) {
		int index = INTERCONNECT_COMPONENT_COUNT * i;

		/*double* source = NULL;
		cout << endl << "Source = NULL" << endl;*/

		int mStart = sendBorderInfo[ index + M_OFFSET ];
		int mStop = mStart + sendBorderInfo[ index + M_LENGTH ];

		int nStart = sendBorderInfo[ index + N_OFFSET ];
		int nStop = nStart + sendBorderInfo[ index + N_LENGTH ];

		switch (sendBorderInfo[index + SIDE]) {
			case LEFT:
				prepareBorder(i, mStart, mStop, nStart, nStop, 0, haloSize);
				break;
			case RIGHT:
				prepareBorder(i, mStart, mStop, nStart, nStop, xCount - haloSize, xCount);
				break;
			case FRONT:
				prepareBorder(i, mStart, mStop, 0, haloSize, nStart, nStop);
				break;
			case BACK:
				prepareBorder(i, mStart, mStop, yCount - haloSize, yCount, nStart, nStop);
				break;
			case TOP:
				prepareBorder(i, 0, haloSize, mStart, mStop, nStart, nStop);
				break;
			case BOTTOM:
				prepareBorder(i, zCount - haloSize, zCount, mStart, mStop, nStart, nStop);
				break;
			default:
				break;
		}
	}
}

void Block::print() {
	cout << "####################################################################################################" << endl;

	printGeneralInformation();
	printSendBorderInfo();
	printReceiveBorderInfo();
	printParameters();
	printComputeFunctionNumber();

	mSolver->print(zCount, yCount, xCount, cellSize);

	cout << "####################################################################################################" << endl;
}

void Block::printGeneralInformation() {
	cout << endl;
	cout << "Block from node #" << nodeNumber << endl;
	cout << "Dimension    " << dimension << endl;
	cout << endl;
	cout << "xCount:      " << xCount << endl;
	cout << "yCount:      " << yCount << endl;
	cout << "zCount:      " << zCount << endl;
	cout << endl;
	cout << "xOffset:     " << xOffset << endl;
	cout << "yOffset:     " << yOffset << endl;
	cout << "zOffset:     " << zOffset << endl;
	cout << endl;
	cout << "Cell size:   " << cellSize << endl;
	cout << "Halo size:   " << haloSize << endl;
}

void Block::printSendBorderInfoArray(int* sendBorderInfoArray) {
	cout << endl;
	cout << "Send border info (" << countSendSegmentBorder << ")" << endl;
	for (int i = 0; i < countSendSegmentBorder; ++i) {
		int index = INTERCONNECT_COMPONENT_COUNT * i;
		cout << "Block border #" << i << endl;
		cout << "	Memory address: " << blockBorder[i] << endl;
		cout << "	Memory type:    " << getMemoryTypeName( blockBorderMemoryAllocType[i] ) << endl;
		cout << "	Side:           " << getSideName( sendBorderInfoArray[index + SIDE] ) << endl;
		cout << "	mOffset:        " << sendBorderInfoArray[index + M_OFFSET] << endl;
		cout << "	nOffset:        " << sendBorderInfoArray[index + N_OFFSET] << endl;
		cout << "	mLength:        " << sendBorderInfoArray[index + M_LENGTH] << endl;
		cout << "	nLength:        " << sendBorderInfoArray[index + N_LENGTH] << endl;
		cout << endl;
	}
}

void Block::printReceiveBorderInfoArray(int* receiveBorderInfoArray) {
	cout << endl << endl;
	cout << "Receive border info (" << countReceiveSegmentBorder << ")" << endl;
	for (int i = 0; i < countReceiveSegmentBorder; ++i) {
		int index = INTERCONNECT_COMPONENT_COUNT * i;
		cout << "Block border #" << i << endl;
		cout << "	Memory address: " << externalBorder[i] << endl;
		cout << "	Memory type:    " << getMemoryTypeName( externalBorderMemoryAllocType[i] ) << endl;
		cout << "	Side:           " << getSideName( receiveBorderInfoArray[index + SIDE] ) << endl;
		cout << "	mOffset:        " << receiveBorderInfoArray[index + M_OFFSET] << endl;
		cout << "	nOffset:        " << receiveBorderInfoArray[index + N_OFFSET] << endl;
		cout << "	mLength:        " << receiveBorderInfoArray[index + M_LENGTH] << endl;
		cout << "	nLength:        " << receiveBorderInfoArray[index + N_LENGTH] << endl;
		cout << endl;
	}
}

void Block::loadData(double* data) {
	mSolver->loadState(data);
}
