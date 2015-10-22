/*
 * realblock.cpp
 *
 *  Created on: 15 окт. 2015 г.
 *      Author: frolov
 */

#include "realblock.h"

RealBlock::RealBlock() {
	// TODO Auto-generated constructor stub

}

RealBlock::~RealBlock() {
	// TODO Auto-generated destructor stub
}

double* RealBlock::getNewBlockBorder(Block* neighbor, int borderLength) {
	//double* tmpBorder;

	if( ( nodeNumber == neighbor->getNodeNumber() ) && neighbor->isProcessingUnitGPU() ) {
		//cudaMallocHost ( (void**)&tmpBorder, borderLength * sizeof(double) );
		//tempBlockBorderMemoryAllocType.push_back(CUDA_MALLOC_HOST);
		//memoryType = CUDA_MALLOC_HOST;
		return pu->newDoublePinnedArray(borderLength);
	}
	else {
		//tmpBorder = new double [borderLength];
		//tempBlockBorderMemoryAllocType.push_back(NEW);
		//memoryType = NEW;
		return pu->newDoubleArray(borderLength);
	}

	//return tmpBorder;
}

double* RealBlock::getNewExternalBorder(Block* neighbor, int borderLength, double* border) {
	//double* tmpBorder;

	if( nodeNumber == neighbor->getNodeNumber() ) {
		//tmpBorder = border;
		//tempExternalBorderMemoryAllocType.push_back(NOT_ALLOC);
		//memoryType = NOT_ALLOC;
		return border;
	}
	else {
		//tmpBorder = new double [borderLength];
		//tempExternalBorderMemoryAllocType.push_back(NEW);
		//memoryType = NEW;
		return pu->newDoubleArray(borderLength);
	}

	//return tmpBorder;
}

void RealBlock::computeStageBorder(int stage, double time) {
	double* result = problem->getResult(stage, time);
	double* source = problem->getSource(stage, time);

	printf("\nsource must be double**. => source - &source. Error here\n");
	//TODO исправить в ProblemType тип возвращаемого значения для getSource
	pu->computeBorder(mUserFuncs, mCompFuncNumber, result, &source, time, mParams, externalBorder, zCount, yCount, xCount, haloSize);
}

void RealBlock::computeStageCenter(int stage, double time) {
	double* result = problem->getResult(stage, time);
	double* source = problem->getSource(stage, time);

	printf("\nsource must be double**. => source - &source. Error here\n");
	//TODO исправить в ProblemType тип возвращаемого значения для getSource
	pu->computeCenter(mUserFuncs, mCompFuncNumber, result, &source, time, mParams, externalBorder, zCount, yCount, xCount, haloSize);
}

void RealBlock::prepareArgument(int stage, double timestep) {
	problem->prepareArgument(pu, stage, timestep);
}

void RealBlock::prepareStageData(int stage) {
	double* source = problem->getCurrentStateStageData(stage);
	for (int i = 0; i < countSendSegmentBorder; ++i) {
		double* result = blockBorder[i];

		int index = INTERCONNECT_COMPONENT_COUNT * i;

		/*double* source = NULL;
		cout << endl << "Source = NULL" << endl;*/

		int mStart = sendBorderInfo[ index + M_OFFSET ];
		int mStop = mStart + sendBorderInfo[ index + M_LENGTH ];

		int nStart = sendBorderInfo[ index + N_OFFSET ];
		int nStop = nStart + sendBorderInfo[ index + N_LENGTH ];
		//cout<<"This is block "<<blockNumber<<"preparing data to send: "<< mStart<<" "<<mStop<<" "<<nStart<<" "<<nStop<<endl;
		//cout<< "side is "<<sendBorderInfo[index + SIDE]<<endl;
		switch (sendBorderInfo[index + SIDE]) {
			case LEFT:
				//prepareBorder(i, stage, mStart, mStop, nStart, nStop, 0, haloSize);
				pu->prepareBorder(result, source, mStart, mStop, nStart, nStop, 0, haloSize, yCount, xCount, cellSize);
				break;
			case RIGHT:
				//prepareBorder(i, stage, mStart, mStop, nStart, nStop, xCount - haloSize, xCount);
				pu->prepareBorder(result, source, mStart, mStop, nStart, nStop, xCount - haloSize, xCount, yCount, xCount, cellSize);
				break;
			case FRONT:
				//prepareBorder(i, stage, mStart, mStop, 0, haloSize, nStart, nStop);
				pu->prepareBorder(result, source, mStart, mStop, 0, haloSize, nStart, nStop, yCount, xCount, cellSize);
				break;
			case BACK:
				//prepareBorder(i, stage, mStart, mStop, yCount - haloSize, yCount, nStart, nStop);
				pu->prepareBorder(result, source, mStart, mStop, yCount - haloSize, yCount, nStart, nStop, yCount, xCount, cellSize);
				break;
			case TOP:
				//prepareBorder(i, stage, 0, haloSize, mStart, mStop, nStart, nStop);
				pu->prepareBorder(result, source, 0, haloSize, mStart, mStop, nStart, nStop, yCount, xCount, cellSize);
				break;
			case BOTTOM:
				//prepareBorder(i, stage, zCount - haloSize, zCount, mStart, mStop, nStart, nStop);
				pu->prepareBorder(result, source, zCount - haloSize, zCount, mStart, mStop, nStart, nStop, yCount, xCount, cellSize);
				break;
			default:
				break;
		}
	}
}

bool RealBlock::isRealBlock() {
	return true;
}

int RealBlock::getBlockType() {
	return pu->getType();
}

bool RealBlock::isProcessingUnitCPU() {
	return pu->isCPU();
}

bool RealBlock::isProcessingUnitGPU() {
	return pu->isGPU();
}

double RealBlock::getStepError(double timestep) {
	return problem->getStepError(pu, timestep);
}

void RealBlock::confirmStep(double timestep) {
	problem->confirmStep(pu, timestep);
}

void RealBlock::rejectStep(double timestep) {
	problem->rejectStep(pu, timestep);
}

double* RealBlock::addNewBlockBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength) {
	countSendSegmentBorder++;

	tempSendBorderInfo.push_back(side);
	tempSendBorderInfo.push_back(mOffset);
	tempSendBorderInfo.push_back(nOffset);
	tempSendBorderInfo.push_back(mLength);
	tempSendBorderInfo.push_back(nLength);

	int borderLength = mLength * nLength * cellSize * haloSize;

	double* newBlockBorder = getNewBlockBorder(neighbor, borderLength);

	/*if( ( nodeNumber == neighbor->getNodeNumber() ) && isGPU( neighbor->getBlockType() ) ) {
		cudaMallocHost ( (void**)&newBlockBorder, borderLength * sizeof(double) );
		tempBlockBorderMemoryAllocType.push_back(CUDA_MALLOC_HOST);
	}
	else {
		newBlockBorder = new double [borderLength];
		tempBlockBorderMemoryAllocType.push_back(NEW);
	}*/

	tempBlockBorder.push_back(newBlockBorder);

	return newBlockBorder;
}

double* RealBlock::addNewExternalBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength, double* border) {
	countReceiveSegmentBorder++;

	tempReceiveBorderInfo.push_back(side);
	tempReceiveBorderInfo.push_back(mOffset);
	tempReceiveBorderInfo.push_back(nOffset);
	tempReceiveBorderInfo.push_back(mLength);
	tempReceiveBorderInfo.push_back(nLength);

	int borderLength = mLength * nLength * cellSize * haloSize;

	double* newExternalBorder = getNewExternalBorder(neighbor, borderLength, border);

	/*if( nodeNumber == neighbor->getNodeNumber() ) {
		newExternalBorder = border;
		tempExternalBorderMemoryAllocType.push_back(NOT_ALLOC);
	}
	else {
		newExternalBorder = new double [borderLength];
		tempExternalBorderMemoryAllocType.push_back(NEW);
	}*/

	tempExternalBorder.push_back(newExternalBorder);

	return newExternalBorder;
}
