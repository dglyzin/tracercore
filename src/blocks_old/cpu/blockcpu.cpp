/*
 * BlockCpu.cpp
 *
 *  Created on: 20 янв. 2015 г.
 *      Author: frolov
 */

#include "../../blocks_old/cpu/blockcpu.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
//#include "userfuncs.h"

using namespace std;

BlockCpu::BlockCpu(int _blockNumber, int _dimension, int _xCount, int _yCount, int _zCount,
		int _xOffset, int _yOffset, int _zOffset,
		int _nodeNumber, int _deviceNumber,
		int _haloSize, int _cellSize,
		unsigned short int* _initFuncNumber, unsigned short int* _compFuncNumber,
		int _solverIdx, double _aTol, double _rTol) :
				Block( _blockNumber, _dimension, _xCount, _yCount, _zCount,
				_xOffset, _yOffset, _zOffset,
				_nodeNumber, _deviceNumber,
				_haloSize, _cellSize) {
	cout << "Creating block "<<blockNumber<<"...\n";

	createSolver(_solverIdx, _aTol, _rTol);

	int count = getGridNodeCount();

	mCompFuncNumber = new unsigned short int [count];

	for (int i = 0; i < count; ++i) {
		mCompFuncNumber[i] = _compFuncNumber[i];
	}

	getFuncArray(&mUserFuncs, blockNumber);
	getInitFuncArray(&mUserInitFuncs);
	initDefaultParams(&mParams, &mParamsCount);


	cout << "Default params ("<<mParamsCount<<"): ";
	for (int idx=0;idx<mParamsCount; idx++)
		cout <<mParams[idx] << " ";
	cout << endl;

	cout << "functions loaded\n";

	//printf("Func array points to %d \n", (long unsigned int) mUserFuncs );

	//mUserFuncs[0](newMatrix, matrix, 0.0, 2, 2, 0, mParams, NULL);
	//printf("Func array points to %d \n", (long unsigned int) mUserInitFuncs );
	double* matrix = mSolver->getStatePtr();
	mUserInitFuncs[blockNumber](matrix,_initFuncNumber);
	cout << "Initial values filled \n";

}

BlockCpu::~BlockCpu() {
	cout<<"Deleting block"<<endl;
	releaseParams(mParams);
	releaseFuncArray(mUserFuncs);
	releaseInitFuncArray(mUserInitFuncs);
	delete mSolver;
	
	if(blockBorder != NULL) {
		for(int i = 0; i < countSendSegmentBorder; i++ )
			freeMemory(blockBorderMemoryAllocType[i], blockBorder[i]);
		
		delete blockBorder;
		delete blockBorderMemoryAllocType;
	}
	
	
	if(externalBorder != NULL) {
		for(int i = 0; i < countReceiveSegmentBorder; i++ )
			freeMemory(externalBorderMemoryAllocType[i], externalBorder[i]);
		
		delete externalBorder;
		delete externalBorderMemoryAllocType;
	}
}

void BlockCpu::getCurrentState(double* result) {
	mSolver->copyState(result);
}

void BlockCpu::printSendBorderInfo() {
	printSendBorderInfoArray(sendBorderInfo);
	for (int i = 0; i < countSendSegmentBorder; ++i) {
		int count = sendBorderInfo[i * INTERCONNECT_COMPONENT_COUNT + M_LENGTH] * sendBorderInfo[i * INTERCONNECT_COMPONENT_COUNT + N_LENGTH] * cellSize;

		printf("\nsend border #%d\n", i);
		for (int j = 0; j < count; ++j) {
			printf("%.2f ", blockBorder[i][j]);
		}

		printf("\n");
	}
}

void BlockCpu::printReceiveBorderInfo() {
	printReceiveBorderInfoArray(receiveBorderInfo);
	for (int i = 0; i < countReceiveSegmentBorder; ++i) {
		int count = receiveBorderInfo[i * INTERCONNECT_COMPONENT_COUNT + M_LENGTH] * receiveBorderInfo[i * INTERCONNECT_COMPONENT_COUNT + N_LENGTH] * cellSize;

		printf("\nrecv border #%d\n", i);
		for (int j = 0; j < count; ++j) {
			printf("%.2f ", externalBorder[i][j]);
		}

		printf("\n");
	}
}

void BlockCpu::printParameters() {
	cout << endl << endl;
	cout << "Parameters (" << mParamsCount << ")" << endl;
	for (int i = 0; i < mParamsCount; ++i) {
		cout << "	parameter #" << i << ":   " << mParams[i] << endl;
	}
}

void BlockCpu::printComputeFunctionNumber() {
	cout << endl << endl;
	cout << "Compute function number" << endl;
	cout.setf(ios::fixed);
	for (int i = 0; i < zCount; ++i) {
		cout << "z = " << i << endl;

		int zShift = xCount * yCount * i;

		for (int j = 0; j < yCount; ++j) {
			int yShift = xCount * j;

			for (int k = 0; k < xCount; ++k) {
				int xShift = k;
				cout << mCompFuncNumber[ zShift + yShift + xShift ] << " ";
			}
			cout << endl;
		}
	}
	cout << endl;
}

void BlockCpu::moveTempBorderVectorToBorderArray() {
	blockBorder = new double* [countSendSegmentBorder];
	blockBorderMemoryAllocType = new int [countSendSegmentBorder];
	sendBorderInfo = new int [INTERCONNECT_COMPONENT_COUNT * countSendSegmentBorder];

	externalBorder = new double* [countReceiveSegmentBorder];
	externalBorderMemoryAllocType = new int [countReceiveSegmentBorder];
	receiveBorderInfo = new int [INTERCONNECT_COMPONENT_COUNT * countReceiveSegmentBorder];

	for (int i = 0; i < countSendSegmentBorder; ++i) {
		blockBorder[i] = tempBlockBorder.at(i);
		blockBorderMemoryAllocType[i] = tempBlockBorderMemoryAllocType.at(i);

		int index = INTERCONNECT_COMPONENT_COUNT * i;
		sendBorderInfo[ index + SIDE ] = tempSendBorderInfo.at(index + 0);
		sendBorderInfo[ index + M_OFFSET ] = tempSendBorderInfo.at(index + 1);
		sendBorderInfo[ index + N_OFFSET ] = tempSendBorderInfo.at(index + 2);
		sendBorderInfo[ index + M_LENGTH ] = tempSendBorderInfo.at(index + 3);
		sendBorderInfo[ index + N_LENGTH ] = tempSendBorderInfo.at(index + 4);
	}

	for (int i = 0; i < countReceiveSegmentBorder; ++i) {
		externalBorder[i] = tempExternalBorder.at(i);
		externalBorderMemoryAllocType[i] = tempExternalBorderMemoryAllocType.at(i);

		int index = INTERCONNECT_COMPONENT_COUNT * i;
		receiveBorderInfo[ index + SIDE ] = tempReceiveBorderInfo.at(index + 0);
		receiveBorderInfo[ index + M_OFFSET ] = tempReceiveBorderInfo.at(index + 1);
		receiveBorderInfo[ index + N_OFFSET ] = tempReceiveBorderInfo.at(index + 2);
		receiveBorderInfo[ index + M_LENGTH ] = tempReceiveBorderInfo.at(index + 3);
		receiveBorderInfo[ index + N_LENGTH ] = tempReceiveBorderInfo.at(index + 4);
	}

	tempBlockBorder.clear();
	tempExternalBorder.clear();
	
	tempBlockBorderMemoryAllocType.clear();
	tempExternalBorderMemoryAllocType.clear();

	tempSendBorderInfo.clear();
	tempReceiveBorderInfo.clear();
}

void BlockCpu::prepareBorder(int borderNumber, int stage, int zStart, int zStop, int yStart, int yStop, int xStart, int xStop) {
	double* source = mSolver->getStageSource(stage);

	int index = 0;
	for (int z = zStart; z < zStop; ++z) {
		int zShift = xCount * yCount * z;

		for (int y = yStart; y < yStop; ++y) {
			int yShift = xCount * y;

			for (int x = xStart; x < xStop; ++x) {
				int xShift = x;

				for (int c = 0; c < cellSize; ++c) {
					int cellShift = c;
					//printf("block %d is preparing border %d, x=%d, y=%d, z=%d, index=%d\n", blockNumber, borderNumber, x,y,z, index);

					blockBorder[borderNumber][index] = source[ (zShift + yShift + xShift)*cellSize + cellShift ];
					index++;
				}
			}
		}
	}
}

void BlockCpu::createSolver(int solverIdx, double _aTol, double _rTol) {
	int count = getGridElementCount();

	switch (solverIdx) {
		case EULER:
			mSolver = new EulerSolverCpu(count, _aTol, _rTol);
			break;
		case RK4:
			mSolver = new RK4SolverCpu(count, _aTol, _rTol);
			break;
		case DP45:
			mSolver = new DP45SolverCpu(count, _aTol, _rTol);
			break;
		default:
			mSolver = new EulerSolverCpu(count, _aTol, _rTol);
			break;
	}
}

double* BlockCpu::getNewBlockBorder(Block* neighbor, int borderLength, int& memoryType) {
	double* tmpBorder;

	if( ( nodeNumber == neighbor->getNodeNumber() ) && isGPU( neighbor->getBlockType() ) ) {
		cudaMallocHost ( (void**)&tmpBorder, borderLength * sizeof(double) );
		//tempBlockBorderMemoryAllocType.push_back(CUDA_MALLOC_HOST);
		memoryType = CUDA_MALLOC_HOST;
	}
	else {
		tmpBorder = new double [borderLength];
		//tempBlockBorderMemoryAllocType.push_back(NEW);
		memoryType = NEW;
	}

	return tmpBorder;
}

double* BlockCpu::getNewExternalBorder(Block* neighbor, int borderLength, double* border, int& memoryType) {
	double* tmpBorder;

	if( nodeNumber == neighbor->getNodeNumber() ) {
		tmpBorder = border;
		//tempExternalBorderMemoryAllocType.push_back(NOT_ALLOC);
		memoryType = NOT_ALLOC;
	}
	else {
		tmpBorder = new double [borderLength];
		//tempExternalBorderMemoryAllocType.push_back(NEW);
		memoryType = NEW;
	}

	return tmpBorder;
}
