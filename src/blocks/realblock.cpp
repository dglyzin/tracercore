/*
 * realblock.cpp
 *
 *  Created on: 15 окт. 2015 г.
 *      Author: frolov
 */

#include "realblock.h"

using namespace std;

RealBlock::RealBlock(int _nodeNumber, int _dimension,
		int _xCount, int _yCount, int _zCount,
		int _xOffset, int _yOffset, int _zOffset,
		int _cellSize, int _haloSize,
		int _blockNumber, ProcessingUnit* _pu,
		unsigned short int* _initFuncNumber, unsigned short int* _compFuncNumber,
		int problemType, int solverType, double aTol, double rTol) :
		Block(_nodeNumber, _dimension, _xCount, _yCount, _zCount, _xOffset, _yOffset, _zOffset, _cellSize, _haloSize) {
	pu = _pu;

	blockNumber = _blockNumber;

	/*int elementCount = getGridElementCount();
	problem = new Ordinary(pu, solverType, elementCount, aTol, rTol);*/
	problem = createProblem(problemType, solverType, aTol, rTol);

	sendBorderInfo = NULL;
	tempSendBorderInfo.clear();

	receiveBorderInfo = NULL;
	tempReceiveBorderInfo.clear();

	blockBorder = NULL;
	tempBlockBorder.clear();

	externalBorder = NULL;
	tempExternalBorder.clear();

	countSendSegmentBorder = countReceiveSegmentBorder = 0;

	int count = getGridNodeCount();

	mCompFuncNumber = pu->newUnsignedShortIntArray(count);
	mInitFuncNumber = pu->newUnsignedShortIntArray(count);

	pu->copyArray(_compFuncNumber, mCompFuncNumber, count);
	pu->copyArray(_initFuncNumber, mInitFuncNumber, count);


	// TODO зачем mParamCount?
	int mParamsCount = 0;
	getFuncArray(&mUserFuncs, blockNumber);
	getInitFuncArray(&mUserInitFuncs);
	initDefaultParams(&mParams, &mParamsCount);

	double* state = problem->getCurrentStatePointer();
	pu->initState(state, mUserInitFuncs, mInitFuncNumber, blockNumber, 0.0);
	//mUserInitFuncs[blockNumber](matrix, mInitFuncNumber);
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

ProblemType* RealBlock::createProblem(int problemType, int solverType, double aTol, double rTol) {
	int elementCount = getGridElementCount();

	switch (problemType) {
		case ORDINARY:
			return new Ordinary(pu, solverType, elementCount, aTol, rTol);

		case DELAY:
			printf("\nDELAY PROBLEM TYPE NOT READY!!!\n");
			return NULL;

		default:
			return new Ordinary(pu, solverType, elementCount, aTol, rTol);
	}
}

void RealBlock::computeStageBorder(int stage, double time) {
	double* result = problem->getResult(stage);
	double** source = problem->getSource(stage);

	//printf("\nsource must be double**. => source - &source. Error here\n");
	//TODO исправить в ProblemType тип возвращаемого значения для getSource
	pu->computeBorder(mUserFuncs, mCompFuncNumber, result, source, time, mParams, externalBorder, zCount, yCount, xCount, haloSize);
}

void RealBlock::computeStageCenter(int stage, double time) {
	double* result = problem->getResult(stage);
	double** source = problem->getSource(stage);

	//printf("\nsource must be double**. => source - &source. Error here\n");
	//TODO исправить в ProblemType тип возвращаемого значения для getSource
	pu->computeCenter(mUserFuncs, mCompFuncNumber, result, source, time, mParams, externalBorder, zCount, yCount, xCount, haloSize);
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

int RealBlock::getDeviceNumber() {
	return pu->getDeviceNumber();
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

	//printf("\nBorder %d %d %d\n", nodeNumber, blockNumber, newBlockBorder);

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

	//printf("\nExternal %d %d %d\n", nodeNumber, blockNumber, newExternalBorder);

	return newExternalBorder;
}

void RealBlock::moveTempBorderVectorToBorderArray() {
	blockBorder = pu->newDoublePointerArray(countSendSegmentBorder);//new double* [countSendSegmentBorder];
	//blockBorderMemoryAllocType = new int [countSendSegmentBorder];
	sendBorderInfo = pu->newIntArray(INTERCONNECT_COMPONENT_COUNT * countSendSegmentBorder);//new int [INTERCONNECT_COMPONENT_COUNT * countSendSegmentBorder];

	externalBorder = pu->newDoublePointerArray(countReceiveSegmentBorder);//new double* [countReceiveSegmentBorder];
	//externalBorderMemoryAllocType = new int [countReceiveSegmentBorder];
	receiveBorderInfo = pu->newIntArray(INTERCONNECT_COMPONENT_COUNT * countReceiveSegmentBorder);//new int [INTERCONNECT_COMPONENT_COUNT * countReceiveSegmentBorder];

	for (int i = 0; i < countSendSegmentBorder; ++i) {
		blockBorder[i] = tempBlockBorder.at(i);
		//blockBorderMemoryAllocType[i] = tempBlockBorderMemoryAllocType.at(i);

		int index = INTERCONNECT_COMPONENT_COUNT * i;
		sendBorderInfo[ index + SIDE ] = tempSendBorderInfo.at(index + 0);
		sendBorderInfo[ index + M_OFFSET ] = tempSendBorderInfo.at(index + 1);
		sendBorderInfo[ index + N_OFFSET ] = tempSendBorderInfo.at(index + 2);
		sendBorderInfo[ index + M_LENGTH ] = tempSendBorderInfo.at(index + 3);
		sendBorderInfo[ index + N_LENGTH ] = tempSendBorderInfo.at(index + 4);
	}

	for (int i = 0; i < countReceiveSegmentBorder; ++i) {
		externalBorder[i] = tempExternalBorder.at(i);
		//externalBorderMemoryAllocType[i] = tempExternalBorderMemoryAllocType.at(i);

		int index = INTERCONNECT_COMPONENT_COUNT * i;
		receiveBorderInfo[ index + SIDE ] = tempReceiveBorderInfo.at(index + 0);
		receiveBorderInfo[ index + M_OFFSET ] = tempReceiveBorderInfo.at(index + 1);
		receiveBorderInfo[ index + N_OFFSET ] = tempReceiveBorderInfo.at(index + 2);
		receiveBorderInfo[ index + M_LENGTH ] = tempReceiveBorderInfo.at(index + 3);
		receiveBorderInfo[ index + N_LENGTH ] = tempReceiveBorderInfo.at(index + 4);
	}

	tempBlockBorder.clear();
	tempExternalBorder.clear();

	//tempBlockBorderMemoryAllocType.clear();
	//tempExternalBorderMemoryAllocType.clear();

	tempSendBorderInfo.clear();
	tempReceiveBorderInfo.clear();
}

void RealBlock::loadData(double* data) {
	problem->loadData(pu, data);
}

void RealBlock::getCurrentState(double* result) {
	problem->getCurrentState(pu, result);
}

void RealBlock::saveState(char* path) {
	problem->saveState(pu, path);
}

void RealBlock::loadState(ifstream& in) {
	problem->loadState(pu, in);
}
