/*
 * BlockGpu.cpp
 *
 *  Created on: 29 янв. 2015 г.
 *      Author: frolov
 */

#include "blockgpu.h"

using namespace std;

BlockGpu::BlockGpu(int _dimension, int _xCount, int _yCount, int _zCount,
		int _xOffset, int _yOffset, int _zOffset,
		int _nodeNumber, int _deviceNumber,
		int _haloSize, int _cellSize,
		unsigned short int* _initFuncNumber, unsigned short int* _compFuncNumber,
		int _mSolverIndex) :
				Block( _dimension, _xCount, _yCount, _zCount,
				_xOffset, _yOffset, _zOffset,
				_nodeNumber, _deviceNumber,
				_haloSize, _cellSize ) {
	
	cudaSetDevice(deviceNumber);
	


	/*dim3 threads ( BLOCK_SIZE );
	dim3 blocksLength  ( (int)ceil((double)length / threads.x) );
	dim3 blocksWidth  ( (int)ceil((double)width / threads.x) );
	dim3 blocksLengthWidth ( (int)ceil((double)(length * width) / threads.x) );
	
	cudaMalloc( (void**)&matrix, width * length * sizeof(double) );
	cudaMalloc( (void**)&newMatrix, width * length * sizeof(double) );
	
	assignDoubleArray <<< blocksLengthWidth, threads >>> ( matrix, 0, length * width);
	assignDoubleArray <<< blocksLengthWidth, threads >>> ( newMatrix, 0, length * width);*/
}

BlockGpu::~BlockGpu() {
	/*if(matrix != NULL)
		cudaFree(matrix);
	
	if(newMatrix != NULL)
		cudaFree(newMatrix);	
	
	if(blockBorder != NULL) {
		for(int i = 0; i < countSendSegmentBorder; i++ )
			freeMemory(blockBorderMemoryAllocType[i], blockBorder[i]);
		
		delete blockBorder;
		cudaFree(blockBorderOnDevice);
		delete blockBorderMemoryAllocType;
	}
	
	if(blockBorderMove != NULL)
		cudaFree(blockBorderMove);
	
	
	if(externalBorder != NULL) {
		for(int i = 0; i < countReceiveSegmentBorder; i++ )
			freeMemory(externalBorderMemoryAllocType[i], externalBorder[i]);
		
		delete externalBorder;
		cudaFree(externalBorderOnDevice);
		delete externalBorderMemoryAllocType;
	}
	
	if(externalBorderMove != NULL)
		cudaFree(externalBorderMove);*/
}

/*void BlockGpu::computeOneStepBorder(double dX2, double dY2, double dT) {
	cudaSetDevice(deviceNumber);
		
	dim3 threads ( BLOCK_LENGHT_SIZE, BLOCK_WIDTH_SIZE );
	dim3 blocks  ( (int)ceil((double)length / threads.x), (int)ceil((double)width / threads.y) );

	calcBorder <<< blocks, threads >>> ( matrix, newMatrix, length, width, dX2, dY2, dT, receiveBorderTypeOnDevice, externalBorderOnDevice, externalBorderMove );
}*/

/*void BlockGpu::computeOneStepCenter(double dX2, double dY2, double dT) {
	cudaSetDevice(deviceNumber);
		
	dim3 threads ( BLOCK_LENGHT_SIZE, BLOCK_WIDTH_SIZE );
	dim3 blocks  ( (int)ceil((double)length / threads.x), (int)ceil((double)width / threads.y) );

	calcCenter <<< blocks, threads >>> ( matrix, newMatrix, length, width, dX2, dY2, dT, receiveBorderTypeOnDevice, externalBorderOnDevice, externalBorderMove );
}*/

/*void BlockGpu::prepareData() {
	cudaSetDevice(deviceNumber);
	
	cudaThreadSynchronize();
	
	dim3 threads ( BLOCK_SIZE );
	dim3 blocksLength  ( (int)ceil((double)length / threads.x) );
	dim3 blocksWidth  ( (int)ceil((double)width / threads.x) );
	
	copyBorderFromMatrix <<< blocksWidth, threads >>> (blockBorderOnDevice, matrix, sendBorderTypeOnDevice, blockBorderMove, TOP, length, width);
	copyBorderFromMatrix <<< blocksLength, threads >>> (blockBorderOnDevice, matrix, sendBorderTypeOnDevice, blockBorderMove, LEFT, length, width);
	copyBorderFromMatrix <<< blocksWidth, threads >>> (blockBorderOnDevice, matrix, sendBorderTypeOnDevice, blockBorderMove, BOTTOM, length, width);
	copyBorderFromMatrix <<< blocksLength, threads >>> (blockBorderOnDevice, matrix, sendBorderTypeOnDevice, blockBorderMove, RIGHT, length, width);
	
	cudaThreadSynchronize();
}*/

void BlockGpu::getCurrentState(double* result) {
	cudaSetDevice(deviceNumber);

	cout << endl << "GPU get cut state maybe bad" << endl;
	//cudaMemcpy( result, matrix, count * sizeof(double), cudaMemcpyDeviceToHost );
}

void BlockGpu::print() {
	cudaSetDevice(deviceNumber);
	
	int* tmpSendBorderInfo = new int [ INTERCONNECT_COMPONENT_COUNT * countSendSegmentBorder ];
	cudaMemcpy( tmpSendBorderInfo, sendBorderInfo, INTERCONNECT_COMPONENT_COUNT * countSendSegmentBorder * sizeof(int), cudaMemcpyDeviceToHost );

	int* tmpReceiveBorderInfo = new int [ INTERCONNECT_COMPONENT_COUNT * countSendSegmentBorder ];
	cudaMemcpy( tmpReceiveBorderInfo, receiveBorderInfo, INTERCONNECT_COMPONENT_COUNT * countReceiveSegmentBorder * sizeof(int), cudaMemcpyDeviceToHost );

	cout << "########################################################################################################################################################################################################" << endl;

	cout << endl;
	cout << "BlockGpu from node #" << nodeNumber << endl;
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

	cout << endl;
	cout << "Block matrix:" << endl;
	cout.setf(ios::fixed);


	// TODO вывод информации о Solver'е

	cout << endl;
	cout << "Send border info (" << countSendSegmentBorder << ")" << endl;
	for (int i = 0; i < countSendSegmentBorder; ++i) {
		int index = INTERCONNECT_COMPONENT_COUNT * i;
		cout << "Block border #" << i << endl;
		cout << "	Memory address: " << blockBorder[i] << endl;
		cout << "	Memory type:    " << getMemoryTypeName( blockBorderMemoryAllocType[i] ) << endl;
		cout << "	Side:           " << getSideName( tmpSendBorderInfo[index + SIDE] ) << endl;
		cout << "	mOffset:        " << tmpSendBorderInfo[index + M_OFFSET] << endl;
		cout << "	nOffset:        " << tmpSendBorderInfo[index + N_OFFSET] << endl;
		cout << "	mLength:        " << tmpSendBorderInfo[index + M_LENGTH] << endl;
		cout << "	nLength:        " << tmpSendBorderInfo[index + N_LENGTH] << endl;
		cout << endl;
	}

	cout << endl << endl;
	cout << "Receive border info (" << countReceiveSegmentBorder << ")" << endl;
	for (int i = 0; i < countReceiveSegmentBorder; ++i) {
		int index = INTERCONNECT_COMPONENT_COUNT * i;
		cout << "Block border #" << i << endl;
		cout << "	Memory address: " << externalBorder[i] << endl;
		cout << "	Memory type:    " << getMemoryTypeName( externalBorderMemoryAllocType[i] ) << endl;
		cout << "	Side:           " << getSideName( tmpReceiveBorderInfo[index + SIDE] ) << endl;
		cout << "	mOffset:        " << tmpReceiveBorderInfo[index + M_OFFSET] << endl;
		cout << "	nOffset:        " << tmpReceiveBorderInfo[index + N_OFFSET] << endl;
		cout << "	mLength:        " << tmpReceiveBorderInfo[index + M_LENGTH] << endl;
		cout << "	nLength:        " << tmpReceiveBorderInfo[index + N_LENGTH] << endl;
		cout << endl;
	}

	cout << endl << "GPU Параметры не выводятся!!!" << endl;
	/*cout << "Parameters (" << mParamsCount << ")" << endl;
	for (int i = 0; i < mParamsCount; ++i) {
		cout << "	parameter #" << i << ":   " << mParams[i] << endl;
	}*/


	cout << endl << "GPU Информация о функциях не выводится!!!" << endl;
	/*cout << "Compute function number" << endl;
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
	}*/
	cout << endl;

	cout << "########################################################################################################################################################################################################" << endl;
	cout << endl << endl;

	delete tmpSendBorderInfo;
	delete tmpReceiveBorderInfo;
}

/*double* BlockGpu::addNewBlockBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength) {
	cudaSetDevice(deviceNumber);

	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)borderLength / threads.x) );
	
	assignIntArray <<< blocks, threads >>> ( sendBorderType[side] + move, countSendSegmentBorder, borderLength );

	countSendSegmentBorder++;

	tempSendBorderInfo.push_back(side);
	tempSendBorderInfo.push_back(mOffset);
	tempSendBorderInfo.push_back(nOffset);
	tempSendBorderInfo.push_back(mLength);
	tempSendBorderInfo.push_back(nLength);

	int borderLength = mLength * nLength * cellSize * haloSize;

	double* newBlockBorder;

	if( nodeNumber == neighbor->getNodeNumber() ) {
		if( isCPU( neighbor->getBlockType() ) ) {
			cudaMallocHost ( (void**)&newBlockBorder, borderLength * sizeof(double) );
			tempBlockBorderMemoryAllocType.push_back(CUDA_MALLOC_HOST);
		}
		
		if( isGPU( neighbor->getBlockType() ) && deviceNumber != neighbor->getDeviceNumber() ) {
			cudaMallocHost ( (void**)&newBlockBorder, borderLength * sizeof(double) );
			tempBlockBorderMemoryAllocType.push_back(CUDA_MALLOC_HOST);
		}
		
		if( isGPU( neighbor->getBlockType() ) && deviceNumber == neighbor->getDeviceNumber() ) {
			cudaMalloc ( (void**)&newBlockBorder, borderLength * sizeof(double) );
			tempBlockBorderMemoryAllocType.push_back(CUDA_MALLOC);
		}
	}
	else {
		cudaMalloc ( (void**)&newBlockBorder, borderLength * sizeof(double) );
		tempBlockBorderMemoryAllocType.push_back(CUDA_MALLOC);
	}

	tempBlockBorder.push_back(newBlockBorder);

	return newBlockBorder;
}*/

/*double* BlockGpu::addNewExternalBorder(Block* neighbor, int side, int move, int borderLength, double* border) {
	cudaSetDevice(deviceNumber);
	
	if( checkValue(side, move + borderLength) ) {
		printf("\nCritical error!\n");
		exit(1);
	}

	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)borderLength / threads.x) );
	
	assignIntArray <<< blocks, threads >>> ( receiveBorderType[side] + move, countReceiveSegmentBorder, borderLength );

	countReceiveSegmentBorder++;

	double* newExternalBorder;

	if( nodeNumber == neighbor->getNodeNumber() ) {
		newExternalBorder = border;
		tempExternalBorderMemoryAllocType.push_back(NOT_ALLOC);
	}
	else {
		cudaMalloc ( (void**)&newExternalBorder, borderLength * sizeof(double) );
		tempExternalBorderMemoryAllocType.push_back(CUDA_MALLOC);
	}

	tempExternalBorder.push_back(newExternalBorder);
	tempExternalBorderMove.push_back(move);

	return newExternalBorder;
}*/

void BlockGpu::moveTempBorderVectorToBorderArray() {
	cudaSetDevice(deviceNumber);
	
	double** tmpBlockBorder = new double* [countSendSegmentBorder];
	cudaMalloc ( (void**)&blockBorder, countSendSegmentBorder * sizeof(double*) );
	int* tmpSendBorderInfo = new int [ INTERCONNECT_COMPONENT_COUNT * countSendSegmentBorder ];
	cudaMalloc ( (void**)&sendBorderInfo, INTERCONNECT_COMPONENT_COUNT * countSendSegmentBorder * sizeof(int) );
	blockBorderMemoryAllocType = new int [countSendSegmentBorder];

	double** tmpExternalBorder = new double* [countReceiveSegmentBorder];
	cudaMalloc ( (void**)&externalBorder, countReceiveSegmentBorder * sizeof(double*) );
	int* tmpReceiveBorderInfo = new int [ INTERCONNECT_COMPONENT_COUNT * countReceiveSegmentBorder ];
	cudaMalloc ( (void**)&receiveBorderInfo, INTERCONNECT_COMPONENT_COUNT * countReceiveSegmentBorder * sizeof(int) );
	externalBorderMemoryAllocType = new int [countReceiveSegmentBorder];	
	

	for (int i = 0; i < countSendSegmentBorder; ++i) {
		tmpBlockBorder[i] = tempBlockBorder.at(i);
		blockBorderMemoryAllocType[i] = tempBlockBorderMemoryAllocType.at(i);

		int index = INTERCONNECT_COMPONENT_COUNT * i;
		tmpSendBorderInfo[ index + SIDE ] = tempSendBorderInfo.at(index + 0);
		tmpSendBorderInfo[ index + M_OFFSET ] = tempSendBorderInfo.at(index + 1);
		tmpSendBorderInfo[ index + N_OFFSET ] = tempSendBorderInfo.at(index + 2);
		tmpSendBorderInfo[ index + M_LENGTH ] = tempSendBorderInfo.at(index + 3);
		tmpSendBorderInfo[ index + N_LENGTH ] = tempSendBorderInfo.at(index + 4);
	}

	for (int i = 0; i < countReceiveSegmentBorder; ++i) {
		tmpExternalBorder[i] = tempExternalBorder.at(i);
		externalBorderMemoryAllocType[i] = tempExternalBorderMemoryAllocType.at(i);

		int index = INTERCONNECT_COMPONENT_COUNT * i;
		tmpReceiveBorderInfo[ index + SIDE ] = tempReceiveBorderInfo.at(index + 0);
		tmpReceiveBorderInfo[ index + M_OFFSET ] = tempReceiveBorderInfo.at(index + 1);
		tmpReceiveBorderInfo[ index + N_OFFSET ] = tempReceiveBorderInfo.at(index + 2);
		tmpReceiveBorderInfo[ index + M_LENGTH ] = tempReceiveBorderInfo.at(index + 3);
		tmpReceiveBorderInfo[ index + N_LENGTH ] = tempReceiveBorderInfo.at(index + 4);
	}

	tempBlockBorder.clear();
	tempExternalBorder.clear();
	
	tempBlockBorderMemoryAllocType.clear();
	tempExternalBorderMemoryAllocType.clear();

	tempSendBorderInfo.clear();
	tempReceiveBorderInfo.clear();

	cudaMemcpy( blockBorder, tmpBlockBorder, countSendSegmentBorder * sizeof(double*), cudaMemcpyHostToDevice );
	
	cudaMemcpy( externalBorder, tmpExternalBorder, countReceiveSegmentBorder * sizeof(double*), cudaMemcpyHostToDevice );
	
	cudaMemcpy( sendBorderInfo, tmpSendBorderInfo, INTERCONNECT_COMPONENT_COUNT * countSendSegmentBorder * sizeof(int), cudaMemcpyHostToDevice );

	cudaMemcpy( receiveBorderInfo, tmpReceiveBorderInfo, INTERCONNECT_COMPONENT_COUNT * countReceiveSegmentBorder * sizeof(int), cudaMemcpyHostToDevice );
	
	delete tmpBlockBorder;
	delete tmpExternalBorder;
	delete tmpSendBorderInfo;
	delete tmpReceiveBorderInfo;
}

void BlockGpu::loadData(double* data) {
	cudaSetDevice(deviceNumber);
	
	cout << endl << "GPU LOAD DATA NOT WORK!" << endl;
	return;
	
	/*cudaMemcpy( matrix, data, sizeof(double) * length * width, cudaMemcpyHostToDevice );*/
}

void BlockGpu::createSolver(int solverIdx) {
	cudaSetDevice(deviceNumber);

	int count = getGridElementCount();

	switch (solverIdx) {
		case EULER:
			mSolver = new EulerSolverGpu(count);
			break;
		case RK4:
			mSolver = new RK4SolverGpu(count);
			break;
		case DP45:
			mSolver = new DP45SolverGpu(count);
			break;
		default:
			mSolver = new EulerSolverGpu(count);
			break;
	}
}

double* BlockGpu::getNewBlockBorder(Block* neighbor, int borderLength, int& memoryType) {
	double* tmpBorder;

	if( nodeNumber == neighbor->getNodeNumber() ) {
		if( isCPU( neighbor->getBlockType() ) ) {
			cudaMallocHost ( (void**)&tmpBorder, borderLength * sizeof(double) );
			//tempBlockBorderMemoryAllocType.push_back(CUDA_MALLOC_HOST);
			memoryType = CUDA_MALLOC_HOST;
		}

		if( isGPU( neighbor->getBlockType() ) && deviceNumber != neighbor->getDeviceNumber() ) {
			cudaMallocHost ( (void**)&tmpBorder, borderLength * sizeof(double) );
			//tempBlockBorderMemoryAllocType.push_back(CUDA_MALLOC_HOST);
			memoryType = CUDA_MALLOC_HOST;
		}

		if( isGPU( neighbor->getBlockType() ) && deviceNumber == neighbor->getDeviceNumber() ) {
			cudaMalloc ( (void**)&tmpBorder, borderLength * sizeof(double) );
			//tempBlockBorderMemoryAllocType.push_back(CUDA_MALLOC);
			memoryType = CUDA_MALLOC;
		}
	}
	else {
		cudaMalloc ( (void**)&tmpBorder, borderLength * sizeof(double) );
		//tempBlockBorderMemoryAllocType.push_back(CUDA_MALLOC);
		memoryType = CUDA_MALLOC;
	}

	return tmpBorder;
}

double* BlockGpu::getNewExternalBorder(Block* neighbor, int borderLength, double* border, int& memoryType) {
	double* tmpBorder;

	if( nodeNumber == neighbor->getNodeNumber() ) {
		tmpBorder = border;
		//tempExternalBorderMemoryAllocType.push_back(NOT_ALLOC);
		memoryType = NOT_ALLOC;
	}
	else {
		cudaMalloc ( (void**)&tmpBorder, borderLength * sizeof(double) );
		//tempExternalBorderMemoryAllocType.push_back(CUDA_MALLOC);
		memoryType = CUDA_MALLOC;
	}

	return tmpBorder;
}
