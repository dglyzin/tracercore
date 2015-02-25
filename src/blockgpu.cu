/*
 * BlockGpu.cpp
 *
 *  Created on: 29 янв. 2015 г.
 *      Author: frolov
 */

#include "blockgpu.h"

/*
 * Функция ядра.
 * Расчет теплоемкости на видеокарте.
 * Логика функции аналогична функции для центрального процессора.
 */
__global__ void calc ( double* matrix, double* newMatrix, int length, int width, double dX2, double dY2, double dT, int **recieveBorderType, double** externalBorder, int* externalBorderMove ) {

	double top, left, bottom, right, cur;

	int i = BLOCK_LENGHT_SIZE * blockIdx.x + threadIdx.x;
	int j = BLOCK_WIDTH_SIZE * blockIdx.y + threadIdx.y;

	if( i < length && j < width ) {
		if( i == 0 )
			if( recieveBorderType[TOP][j] == BY_FUNCTION ) {
				newMatrix[i * width + j] = 100;
				return;
			}
			else
				top = externalBorder[	recieveBorderType[TOP][j]	][j - externalBorderMove[	recieveBorderType[TOP][j]	]];
		else
			top = matrix[(i - 1) * width + j];
	
	
		if( j == 0 )
			if( recieveBorderType[LEFT][i] == BY_FUNCTION ) {
				newMatrix[i * width + j] = 10;
				return;
			}
			else
				left = externalBorder[	recieveBorderType[LEFT][i]	][i - externalBorderMove[	recieveBorderType[LEFT][i]		]];
		else
			left = matrix[i * width + (j - 1)];
	
	
		if( i == length - 1 )
			if( recieveBorderType[BOTTOM][j] == BY_FUNCTION ) {
				newMatrix[i * width + j] = 10;
				return;
			}
			else
				bottom = externalBorder[	recieveBorderType[BOTTOM][j]	][j - externalBorderMove[	recieveBorderType[BOTTOM][j]	]];
		else
			bottom = matrix[(i + 1) * width + j];
	
	
		if( j == width - 1 )
			if( recieveBorderType[RIGHT][i] == BY_FUNCTION ) {
				newMatrix[i * width + j] = 10;
				return;
			}
			else
				right = externalBorder[	recieveBorderType[RIGHT][i]	][i - externalBorderMove[	recieveBorderType[RIGHT][i]	]];
		else
			right = matrix[i * width + (j + 1)];
	
	
		cur = matrix[i * width + j];
	
		newMatrix[i * width + j] = cur + dT * ( ( left - 2*cur + right )/dX2 + ( top - 2*cur + bottom )/dY2  );
	}
}

/*
 * Функция ядра
 * Заполнение целочисленного массива определенным значением.
 */
__global__ void assignIntArray (int* arr, int value, int arrayLength) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < arrayLength )
		arr[idx] = value;
}

/*
 * Функция ядра
 * Копирование целочесленных массивов.
 */
__global__ void copyIntArray (int* dest, int* source, int arrayLength) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < arrayLength )
		dest[idx] = source[idx];
}

/*
 * Функция ядра
 * Заполнение вещественного массива определенным значением.
 */
__global__ void assignDoubleArray (double* arr, double value, int arrayLength) {
	int	idx = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( idx < arrayLength )
		arr[idx] = value;
}

/*
 * Функция ядра
 * Копирование данных из матрицы в массив.
 * Используется при подготовке пересылаемых данных.
 */
__global__ void copyBorderFromMatrix ( double** blockBorder, double* matrix, int** sendBorderType, int* blockBorderMove, int side, int length, int width )
{
	int idx  = BLOCK_SIZE * blockIdx.x + threadIdx.x;
	
	if( (side == TOP || side == BOTTOM) && idx >= width )
		return;
	
	if( (side == LEFT || side == RIGHT) && idx >= length )
		return;

	if( sendBorderType[side][idx] == BY_FUNCTION )
		return;
	
	double value;
	
	switch (side) {
		case TOP:
			value = matrix[0 * width + idx];
			break;
		case LEFT:
			value = matrix[idx * width + 0];
			break;
		case BOTTOM:
			value = matrix[(length - 1) * width + idx];
			break;
		case RIGHT:
			value = matrix[idx * width + (width - 1)];
			break;
		default:
			break;
	}
	
	blockBorder[	sendBorderType[side][idx]	][idx - blockBorderMove[	sendBorderType[side][idx]	]] = value;
}

BlockGpu::BlockGpu(int _length, int _width, int _lengthMove, int _widthMove, int _nodeNumber, int _deviceNumber) : Block(  _length, _width, _lengthMove, _widthMove, _nodeNumber, _deviceNumber ) {
	deviceNumber = _deviceNumber;
	
	cudaSetDevice(deviceNumber);
	
	dim3 threads ( BLOCK_SIZE );
	dim3 blocksLength  ( (int)ceil((double)length / threads.x) );
	dim3 blocksWidth  ( (int)ceil((double)width / threads.x) );
	dim3 blocksLengthWidth ( (int)ceil((double)(length * width) / threads.x) );
	
	cudaMalloc( (void**)&matrix, width * length * sizeof(double) );
	cudaMalloc( (void**)&newMatrix, width * length * sizeof(double) );
	
	assignDoubleArray <<< blocksLengthWidth, threads >>> ( matrix, 0, length * width);
	assignDoubleArray <<< blocksLengthWidth, threads >>> ( newMatrix, 0, length * width);

	/*
	 * Типы границ блока. Выделение памяти.
	 */
	sendBorderType = new int* [BORDER_COUNT];

	cudaMalloc ( (void**)&sendBorderType[TOP], width * sizeof(int) );
	assignIntArray <<< blocksWidth, threads >>> ( sendBorderType[TOP], BY_FUNCTION, width ); 

	cudaMalloc ( (void**)&sendBorderType[LEFT], length * sizeof(int) );
	assignIntArray <<< blocksLength, threads >>> ( sendBorderType[LEFT], BY_FUNCTION, length );

	cudaMalloc ( (void**)&sendBorderType[BOTTOM], width * sizeof(int) );
	assignIntArray <<< blocksWidth, threads >>> ( sendBorderType[BOTTOM], BY_FUNCTION, width ); 

	cudaMalloc ( (void**)&sendBorderType[RIGHT], length * sizeof(int) );
	assignIntArray <<< blocksLength, threads >>> ( sendBorderType[RIGHT], BY_FUNCTION, length );
	
	cudaMalloc ( (void**)&sendBorderTypeOnDevice, BORDER_COUNT * sizeof(int*) );
	cudaMemcpy( sendBorderTypeOnDevice, sendBorderType, BORDER_COUNT * sizeof(int*), cudaMemcpyHostToDevice );
	
	
	receiveBorderType = new int* [BORDER_COUNT];

	cudaMalloc ( (void**)&receiveBorderType[TOP], width * sizeof(int) );
	assignIntArray <<< blocksWidth, threads >>> ( receiveBorderType[TOP], BY_FUNCTION, width ); 

	cudaMalloc ( (void**)&receiveBorderType[LEFT], length * sizeof(int) );
	assignIntArray <<< blocksLength, threads >>> ( receiveBorderType[LEFT], BY_FUNCTION, length );

	cudaMalloc ( (void**)&receiveBorderType[BOTTOM], width * sizeof(int) );
	assignIntArray <<< blocksWidth, threads >>> ( receiveBorderType[BOTTOM], BY_FUNCTION, width ); 

	cudaMalloc ( (void**)&receiveBorderType[RIGHT], length * sizeof(int) );
	assignIntArray <<< blocksLength, threads >>> ( receiveBorderType[RIGHT], BY_FUNCTION, length );
	
	cudaMalloc ( (void**)&receiveBorderTypeOnDevice, BORDER_COUNT * sizeof(int*) );
	cudaMemcpy( receiveBorderTypeOnDevice, receiveBorderType, BORDER_COUNT * sizeof(int*), cudaMemcpyHostToDevice );
	
	result = new double [length * width];
}

BlockGpu::~BlockGpu() {
	if(matrix != NULL)
		cudaFree(matrix);
	
	if(newMatrix != NULL)
		cudaFree(newMatrix);
	
	if(sendBorderType != NULL) {
		if(sendBorderType[TOP] != NULL)
			cudaFree(sendBorderType[TOP]);
		
		if(sendBorderType[LEFT] != NULL)
			cudaFree(sendBorderType[LEFT]);
		
		if(sendBorderType[BOTTOM] != NULL)
			cudaFree(sendBorderType[BOTTOM]);
		
		if(sendBorderType[RIGHT] != NULL)
			cudaFree(sendBorderType[RIGHT]);
		
		cudaFree(sendBorderTypeOnDevice);
		delete sendBorderType;
	}
	
	if(receiveBorderType != NULL) {
		if(receiveBorderType[TOP] != NULL)
			cudaFree(receiveBorderType[TOP]);
		
		if(receiveBorderType[LEFT] != NULL)
			cudaFree(receiveBorderType[LEFT]);
		
		if(receiveBorderType[BOTTOM] != NULL)
			cudaFree(receiveBorderType[BOTTOM]);
		
		if(receiveBorderType[RIGHT] != NULL)
			cudaFree(receiveBorderType[RIGHT]);
		
		cudaFree(receiveBorderTypeOnDevice);
		delete receiveBorderType;
	}
	
	if(result != NULL)
		delete result;
}

void BlockGpu::computeOneStep(double dX2, double dY2, double dT) {
	cudaSetDevice(deviceNumber);
	
	dim3 threads ( BLOCK_LENGHT_SIZE, BLOCK_WIDTH_SIZE );
	dim3 blocks  ( (int)ceil((double)length / threads.x), (int)ceil((double)width / threads.y) );

	calc <<< blocks, threads >>> ( matrix, newMatrix, length, width, dX2, dY2, dT, receiveBorderTypeOnDevice, externalBorderOnDevice, externalBorderMove );
	
	double* tmp = matrix;

	matrix = newMatrix;

	newMatrix = tmp;
}

void BlockGpu::prepareData() {
	cudaSetDevice(deviceNumber);
	
	dim3 threads ( BLOCK_SIZE );
	dim3 blocksLength  ( (int)ceil((double)length / threads.x) );
	dim3 blocksWidth  ( (int)ceil((double)width / threads.x) );
	
	copyBorderFromMatrix <<< blocksWidth, threads >>> (blockBorderOnDevice, matrix, sendBorderTypeOnDevice, blockBorderMove, TOP, length, width);
	copyBorderFromMatrix <<< blocksLength, threads >>> (blockBorderOnDevice, matrix, sendBorderTypeOnDevice, blockBorderMove, LEFT, length, width);
	copyBorderFromMatrix <<< blocksWidth, threads >>> (blockBorderOnDevice, matrix, sendBorderTypeOnDevice, blockBorderMove, BOTTOM, length, width);
	copyBorderFromMatrix <<< blocksLength, threads >>> (blockBorderOnDevice, matrix, sendBorderTypeOnDevice, blockBorderMove, RIGHT, length, width);
}

double* BlockGpu::getResult() {
	cudaSetDevice(deviceNumber);
	
	cudaMemcpy( result, matrix, width * length * sizeof(double), cudaMemcpyDeviceToHost );
	
	return result;
}

void BlockGpu::print() {
	cudaSetDevice(deviceNumber);
	
	double* matrixToPrint = new double [length * width];
	
	int** sendBorderTypeToPrint = new int* [BORDER_COUNT];
	sendBorderTypeToPrint[TOP] = new int [width];
	sendBorderTypeToPrint[LEFT] = new int [length];
	sendBorderTypeToPrint[BOTTOM] = new int [width];
	sendBorderTypeToPrint[RIGHT] = new int [length];
	
	int** receiveBorderTypeToPrint = new int* [BORDER_COUNT];
	receiveBorderTypeToPrint[TOP] = new int [width];
	receiveBorderTypeToPrint[LEFT] = new int [length];
	receiveBorderTypeToPrint[BOTTOM] = new int [width];
	receiveBorderTypeToPrint[RIGHT] = new int [length];
	
	int* blockBorderMoveToPrint = new int [countSendSegmentBorder];
	int* externalBorderMoveToPrint = new int [countReceiveSegmentBorder];
	
	
	cudaMemcpy( matrixToPrint, matrix, length * width * sizeof(double), cudaMemcpyDeviceToHost );
	
	cudaMemcpy( sendBorderTypeToPrint[TOP], sendBorderType[TOP], width * sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( sendBorderTypeToPrint[LEFT], sendBorderType[LEFT], length * sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( sendBorderTypeToPrint[BOTTOM], sendBorderType[BOTTOM], width * sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( sendBorderTypeToPrint[RIGHT], sendBorderType[RIGHT], length * sizeof(int), cudaMemcpyDeviceToHost );
	
	cudaMemcpy( receiveBorderTypeToPrint[TOP], receiveBorderType[TOP], width * sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( receiveBorderTypeToPrint[LEFT], receiveBorderType[LEFT], length * sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( receiveBorderTypeToPrint[BOTTOM], receiveBorderType[BOTTOM], width * sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( receiveBorderTypeToPrint[RIGHT], receiveBorderType[RIGHT], length * sizeof(int), cudaMemcpyDeviceToHost );
	
	cudaMemcpy( blockBorderMoveToPrint, blockBorderMove, countSendSegmentBorder * sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( externalBorderMoveToPrint, externalBorderMove, countReceiveSegmentBorder * sizeof(int), cudaMemcpyDeviceToHost );
	
	
	printf("FROM NODE #%d", nodeNumber);

	printf("\nLength: %d, Width: %d\n", length, width);
	printf("\nlengthMove: %d, widthMove: %d\n", lenghtMove, widthMove);

	printf("\nMatrix:\n");
	for (int i = 0; i < length; ++i)
	{
		for (int j = 0; j < width; ++j)
			printf("%6.1f ", matrixToPrint[i * width + j]);
		printf("\n");
	}
	
	printf("\ntopSendBorderType\n");
	for (int i = 0; i < width; ++i)
		printf("%4d", sendBorderTypeToPrint[TOP][i]);
	printf("\n");

	printf("\nleftSendBorderType\n");
	for (int i = 0; i < length; ++i)
		printf("%4d", sendBorderTypeToPrint[LEFT][i]);
	printf("\n");

	printf("\nbottomSendBorderType\n");
	for (int i = 0; i < width; ++i)
		printf("%4d", sendBorderTypeToPrint[BOTTOM][i]);
	printf("\n");

	printf("\nrightSendBorderType\n");
	for (int i = 0; i < length; ++i)
		printf("%4d", sendBorderTypeToPrint[RIGHT][i]);
	printf("\n\n\n");
	
	
	
	printf("\ntopReceiveBorderType\n");
	for (int i = 0; i < width; ++i)
		printf("%4d", receiveBorderTypeToPrint[TOP][i]);
	printf("\n");

	printf("\nleftReceiveBorderType\n");
	for (int i = 0; i < length; ++i)
		printf("%4d", receiveBorderTypeToPrint[LEFT][i]);
	printf("\n");

	printf("\nbottomReceiveBorderType\n");
	for (int i = 0; i < width; ++i)
		printf("%4d", receiveBorderTypeToPrint[BOTTOM][i]);
	printf("\n");

	printf("\nrightReceiveBorderType\n");
	for (int i = 0; i < length; ++i)
		printf("%4d", receiveBorderTypeToPrint[RIGHT][i]);
	printf("\n");
	
	
	
	for (int i = 0; i < countSendSegmentBorder; ++i)
		printf("\nblockBorder #%d : %d : %d\n", i, blockBorder[i], blockBorderMoveToPrint[i]);

	for (int i = 0; i < countReceiveSegmentBorder; ++i)
		printf("\nexternalBorder #%d : %d : %d\n", i, externalBorder[i], externalBorderMoveToPrint[i]);
	

	printf("\n\n\n");
}

double* BlockGpu::addNewBlockBorder(Block* neighbor, int side, int move, int borderLength) {
	cudaSetDevice(deviceNumber);
	
	if( checkValue(side, move + borderLength) ) {
		printf("\nCritical error!\n");
		exit(1);
	}

	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)borderLength / threads.x) );
	
	assignIntArray <<< blocks, threads >>> ( sendBorderType[side] + move, countSendSegmentBorder, borderLength );

	countSendSegmentBorder++;

	double* newBlockBorder;

	if( nodeNumber == neighbor->getNodeNumber() ) {
		if( isCPU( neighbor->getBlockType() ) )
			cudaMallocHost ( (void**)&newBlockBorder, borderLength * sizeof(double) );
		
		if( isGPU( neighbor->getBlockType() ) && deviceNumber != neighbor->getDeviceNumber() )
			cudaMallocHost ( (void**)&newBlockBorder, borderLength * sizeof(double) );
	}
	else
		cudaMalloc ( (void**)&newBlockBorder, borderLength * sizeof(double) );

	tempBlockBorder.push_back(newBlockBorder);
	tempBlockBorderMove.push_back(move);

	return newBlockBorder;
}

double* BlockGpu::addNewExternalBorder(Block* neighbor, int side, int move, int borderLength, double* border) {
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

	if( nodeNumber == neighbor->getNodeNumber() )
		newExternalBorder = border;
	else
		cudaMalloc ( (void**)&newExternalBorder, borderLength * sizeof(double) );

	tempExternalBorder.push_back(newExternalBorder);
	tempExternalBorderMove.push_back(move);

	return newExternalBorder;
}

void BlockGpu::moveTempBorderVectorToBorderArray() {
	cudaSetDevice(deviceNumber);
	
	blockBorder = new double* [countSendSegmentBorder];
	int* tempBlockBorderMoveArray = new int [countSendSegmentBorder];

	externalBorder = new double* [countReceiveSegmentBorder];
	int* tempExternalBorderMoveArray = new int [countReceiveSegmentBorder];

	for (int i = 0; i < countSendSegmentBorder; ++i) {
		blockBorder[i] = tempBlockBorder.at(i);
		tempBlockBorderMoveArray[i] = tempBlockBorderMove.at(i);
	}

	for (int i = 0; i < countReceiveSegmentBorder; ++i) {
		externalBorder[i] = tempExternalBorder.at(i);
		tempExternalBorderMoveArray[i] = tempExternalBorderMove.at(i);
	}

	tempBlockBorder.clear();
	tempBlockBorderMove.clear();
	tempExternalBorder.clear();
	tempExternalBorderMove.clear();
	
	cudaMalloc ( (void**)&blockBorderOnDevice, countSendSegmentBorder * sizeof(double*) );
	cudaMemcpy( blockBorderOnDevice, blockBorder, countSendSegmentBorder * sizeof(double*), cudaMemcpyHostToDevice );
	
	cudaMalloc ( (void**)&externalBorderOnDevice, countReceiveSegmentBorder * sizeof(double*) );
	cudaMemcpy( externalBorderOnDevice, externalBorder, countReceiveSegmentBorder * sizeof(double*), cudaMemcpyHostToDevice );
	
	cudaMalloc ( (void**)&blockBorderMove, countSendSegmentBorder * sizeof(int) );
	cudaMemcpy( blockBorderMove, tempBlockBorderMoveArray, countSendSegmentBorder * sizeof(int), cudaMemcpyHostToDevice );
	
	cudaMalloc ( (void**)&externalBorderMove, countReceiveSegmentBorder * sizeof(int) );	
	cudaMemcpy( externalBorderMove, tempExternalBorderMoveArray, countReceiveSegmentBorder * sizeof(int), cudaMemcpyHostToDevice );
	
	delete tempBlockBorderMoveArray;
	delete tempExternalBorderMoveArray;
}