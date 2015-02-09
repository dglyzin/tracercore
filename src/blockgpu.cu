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
__global__ void calc ( double* matrix, double* newMatrix, int length, int width, double dX2, double dY2, double dT, int **borderType, double** externalBorder ) {

	double top, left, bottom, right, cur;

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int i = BLOCK_LENGHT_SIZE * bx + tx;
	int j = BLOCK_WIDTH_SIZE * by + ty;

	if( i >= length || j >= width )
		return;

	if( i == 0 )
		if( borderType[TOP][j] == BY_FUNCTION ) {
			newMatrix[i * width + j] = externalBorder[TOP][j];
			return;
		}
		else
			top = externalBorder[TOP][j];
	else
		top = matrix[(i - 1) * width + j];


	if( j == 0 )
		if( borderType[LEFT][i] == BY_FUNCTION ) {
			newMatrix[i * width + j] = externalBorder[LEFT][i];
			return;
		}
		else
			left = externalBorder[LEFT][i];
	else
		left = matrix[i * width + (j - 1)];


	if( i == length - 1 )
		if( borderType[BOTTOM][j] == BY_FUNCTION ) {
			newMatrix[i * width + j] = externalBorder[BOTTOM][j];
			return;
		}
		else
			bottom = externalBorder[BOTTOM][j];
	else
		bottom = matrix[(i + 1) * width + j];


	if( j == width - 1 )
		if( borderType[RIGHT][i] == BY_FUNCTION ) {
			newMatrix[i * width + j] = externalBorder[RIGHT][i];
			return;
		}
		else
			right = externalBorder[RIGHT][i];
	else
		right = matrix[i * width + (j + 1)];


	cur = matrix[i * width + j];

	newMatrix[i * width + j] = cur + dT * ( ( left - 2*cur + right )/dX2 + ( top - 2*cur + bottom )/dY2  );
}

/*
 * Функция ядра
 * Заполнение целочисленного массива определенным значением.
 */
__global__ void assignIntArray (int* arr, int value, int arrayLength) {
	int bx  = blockIdx.x;
	int tx  = threadIdx.x;
	int	idx = BLOCK_SIZE * bx + tx;
	
	if( idx < arrayLength )
		arr[idx] = value;
}

/*
 * Функция ядра
 * Заполнение вещественного массива определенным значением.
 */
__global__ void assignDoubleArray (double* arr, double value, int arrayLength) {
	int bx  = blockIdx.x;
	int tx  = threadIdx.x;
	int	idx = BLOCK_SIZE * bx + tx;
	
	if( idx < arrayLength )
		arr[idx] = value;
}

/*
 * Функция ядра
 * Копирование данных из матрицы в массив.
 * Используется при подготовке пересылаемых данных.
 */
__global__ void copyBorderFromMatrix ( double* dest, double* matrix, int side, int length, int width )
{
	int bx  = blockIdx.x;
	int tx  = threadIdx.x;
	int idx  = BLOCK_SIZE * bx + tx;
	
	if( (side == TOP || side == BOTTOM) && idx >= width )
		return;
	
	if( (side == LEFT || side == RIGHT) && idx >= length )
		return;
	
	switch (side) {
		case TOP:
			dest[idx] = matrix[0 * width + idx];
			break;
		case LEFT:
			dest[idx] = matrix[idx * width + 0];
			break;
		case BOTTOM:
			dest[idx] = matrix[(length - 1) * width + idx];
			break;
		case RIGHT:
			dest[idx] = matrix[idx * width + (width - 1)];
			break;
		default:
			break;
	}
}

BlockGpu::BlockGpu(int _length, int _width, int _lengthMove, int _widthMove, int _world_rank, int _deviceNumber) : Block(  _length, _width, _lengthMove, _widthMove, _world_rank ) {
	deviceNumber = _deviceNumber;
	
	cudaSetDevice(deviceNumber);
	
	dim3 threads ( BLOCK_SIZE );
	dim3 blocksLength  ( (int)ceil((double)length / threads.x) );
	dim3 blocksWidth  ( (int)ceil((double)width / threads.x) );
	dim3 blocksLengthWidth ( (int)ceil((double)(length * width) / threads.x) );
	
	cudaMalloc( (void**)&matrix, width * length * sizeof(double) );
	cudaMalloc( (void**)&newMatrix, width * length * sizeof(double) );
	
	assignDoubleArray <<< blocksLengthWidth, threads >>> ( matrix, 10, length * width);
	assignDoubleArray <<< blocksLengthWidth, threads >>> ( newMatrix, 5, length * width);

	/*
	 * Типы границ блока. Выделение памяти.
	 */
	borderType = new int* [BORDER_COUNT];

	cudaMalloc ( (void**)&borderType[TOP], width * sizeof(int) );
	assignIntArray <<< blocksWidth, threads >>> ( borderType[TOP], BY_FUNCTION, width ); 

	cudaMalloc ( (void**)&borderType[LEFT], length * sizeof(int) );
	assignIntArray <<< blocksLength, threads >>> ( borderType[LEFT], BY_FUNCTION, length );

	cudaMalloc ( (void**)&borderType[BOTTOM], width * sizeof(int) );
	assignIntArray <<< blocksWidth, threads >>> ( borderType[BOTTOM], BY_FUNCTION, width ); 

	cudaMalloc ( (void**)&borderType[RIGHT], length * sizeof(int) );
	assignIntArray <<< blocksLength, threads >>> ( borderType[RIGHT], BY_FUNCTION, length );
	
	cudaMalloc ( (void**)&borderTypeOnDevice, BORDER_COUNT * sizeof(int*) );
	cudaMemcpy( borderTypeOnDevice, borderType, BORDER_COUNT * sizeof(int*), cudaMemcpyHostToDevice );
	
	/*
	 * Границы самого блока.
	 * Это он будет отдавать. Выделение памяти.
	 */
	blockBorder = new double* [BORDER_COUNT];

	cudaMalloc ( (void**)&blockBorder[TOP], width * sizeof(double) );
	assignDoubleArray <<< blocksWidth, threads >>> ( blockBorder[TOP], 0, width );

	cudaMalloc ( (void**)&blockBorder[LEFT], length * sizeof(double) );
	assignDoubleArray <<< blocksLength, threads >>> ( blockBorder[LEFT], 0, length );

	cudaMalloc ( (void**)&blockBorder[BOTTOM], width * sizeof(double) );
	assignDoubleArray <<< blocksWidth, threads >>> ( blockBorder[BOTTOM], 0, width );

	cudaMalloc ( (void**)&blockBorder[RIGHT], length * sizeof(double) );
	assignDoubleArray <<< blocksLength, threads >>> ( blockBorder[RIGHT], 0, length );
	
	cudaMalloc ( (void**)&blockBorderOnDevice, BORDER_COUNT * sizeof(double*) );
	cudaMemcpy( blockBorderOnDevice, blockBorder, BORDER_COUNT * sizeof(double*), cudaMemcpyHostToDevice );


	/*
	 * Внешние границы блока.
	 * Сюда будет приходить информация.
	 */
	externalBorder = new double* [BORDER_COUNT];

	cudaMalloc ( (void**)&externalBorder[TOP], width * sizeof(double) );
	assignDoubleArray <<< blocksWidth, threads >>> ( externalBorder[TOP], 100, width );

	cudaMalloc ( (void**)&externalBorder[LEFT], length * sizeof(double) );
	assignDoubleArray <<< blocksLength, threads >>> ( externalBorder[LEFT], 10, length );

	cudaMalloc ( (void**)&externalBorder[BOTTOM], width * sizeof(double) );
	assignDoubleArray <<< blocksWidth, threads >>> ( externalBorder[BOTTOM], 10, width );

	cudaMalloc ( (void**)&externalBorder[RIGHT], length * sizeof(double) );
	assignDoubleArray <<< blocksLength, threads >>> ( externalBorder[RIGHT], 10, length );
	
	cudaMalloc ( (void**)&externalBorderOnDevice, BORDER_COUNT * sizeof(double*) );
	cudaMemcpy( externalBorderOnDevice, externalBorder, BORDER_COUNT * sizeof(double*), cudaMemcpyHostToDevice );
}

BlockGpu::~BlockGpu() {
	// TODO Auto-generated destructor stub
}

void BlockGpu::courted(double dX2, double dY2, double dT) {
	cudaSetDevice(deviceNumber);
	
	dim3 threads ( BLOCK_LENGHT_SIZE, BLOCK_WIDTH_SIZE );
	dim3 blocks  ( (int)ceil((double)length / threads.x), (int)ceil((double)width / threads.y) );

	calc <<< blocks, threads >>> ( matrix, newMatrix, length, width, dX2, dY2, dT, borderTypeOnDevice, externalBorderOnDevice );
	
	double* tmp = matrix;

	matrix = newMatrix;

	newMatrix = tmp;
}

void BlockGpu::setPartBorder(int type, int side, int move, int borderLength) {
	cudaSetDevice(deviceNumber);
	
	if( checkValue(side, move + borderLength) ) {
		printf("\nCritical error!\n");
		exit(1);
	}
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)borderLength / threads.x) );
	
	assignIntArray <<< blocks, threads >>> ( borderType[side] + move, type, borderLength );
}

void BlockGpu::prepareData() {
	cudaSetDevice(deviceNumber);
	
	dim3 threads ( BLOCK_SIZE );
	dim3 blocksLength  ( (int)ceil((double)length / threads.x) );
	dim3 blocksWidth  ( (int)ceil((double)width / threads.x) );
	
	copyBorderFromMatrix <<< blocksWidth, threads >>> (blockBorder[TOP], matrix, TOP, length, width);
	copyBorderFromMatrix <<< blocksLength, threads >>> (blockBorder[LEFT], matrix, LEFT, length, width);
	copyBorderFromMatrix <<< blocksWidth, threads >>> (blockBorder[BOTTOM], matrix, BOTTOM, length, width);
	copyBorderFromMatrix <<< blocksLength, threads >>> (blockBorder[RIGHT], matrix, RIGHT, length, width);
}

int BlockGpu::getBlockType() {
	switch (deviceNumber) {
		case 0:
			return DEVICE0;
		case 1:
			return DEVICE1;
		case 2:
			return DEVICE2;
		default:
			return DEVICE0;
	}
}

double* BlockGpu::getResault() {
	cudaSetDevice(deviceNumber);
	
	double* res = new double [length * width];
	cudaMemcpy( res, matrix, width * length * sizeof(double), cudaMemcpyDeviceToHost );
	
	return res;
}

void BlockGpu::print() {
	double* matrixToPrint = new double [length * width];
	
	int** borderTypeToPrint = new int* [BORDER_COUNT];
	borderTypeToPrint[TOP] = new int [width];
	borderTypeToPrint[LEFT] = new int [length];
	borderTypeToPrint[BOTTOM] = new int [width];
	borderTypeToPrint[RIGHT] = new int [length];
	
	double** blockBorderToPrint = new double* [BORDER_COUNT];
	blockBorderToPrint[TOP] = new double [width];
	blockBorderToPrint[LEFT] = new double [length];
	blockBorderToPrint[BOTTOM] = new double [width];
	blockBorderToPrint[RIGHT] = new double [length];
	
	double** externalBorderToPrint = new double* [BORDER_COUNT];
	externalBorderToPrint[TOP] = new double [width];
	externalBorderToPrint[LEFT] = new double [length];
	externalBorderToPrint[BOTTOM] = new double [width];
	externalBorderToPrint[RIGHT] = new double [length];
	
	
	cudaMemcpy( matrixToPrint, matrix, length * width * sizeof(double), cudaMemcpyDeviceToHost );
	
	cudaMemcpy( borderTypeToPrint[TOP], borderType[TOP], width * sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( borderTypeToPrint[LEFT], borderType[LEFT], length * sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( borderTypeToPrint[BOTTOM], borderType[BOTTOM], width * sizeof(int), cudaMemcpyDeviceToHost );
	cudaMemcpy( borderTypeToPrint[RIGHT], borderType[RIGHT], length * sizeof(int), cudaMemcpyDeviceToHost );
	
	cudaMemcpy( blockBorderToPrint[TOP], blockBorder[TOP], width * sizeof(double), cudaMemcpyDeviceToHost );
	cudaMemcpy( blockBorderToPrint[LEFT], blockBorder[LEFT], length * sizeof(double), cudaMemcpyDeviceToHost );
	cudaMemcpy( blockBorderToPrint[BOTTOM], blockBorder[BOTTOM], width * sizeof(double), cudaMemcpyDeviceToHost );
	cudaMemcpy( blockBorderToPrint[RIGHT], blockBorder[RIGHT], length * sizeof(double), cudaMemcpyDeviceToHost );
	
	cudaMemcpy( externalBorderToPrint[TOP], externalBorder[TOP], width * sizeof(double), cudaMemcpyDeviceToHost );
	cudaMemcpy( externalBorderToPrint[LEFT], externalBorder[LEFT], length * sizeof(double), cudaMemcpyDeviceToHost );
	cudaMemcpy( externalBorderToPrint[BOTTOM], externalBorder[BOTTOM], width * sizeof(double), cudaMemcpyDeviceToHost );
	cudaMemcpy( externalBorderToPrint[RIGHT], externalBorder[RIGHT], length * sizeof(double), cudaMemcpyDeviceToHost );
	
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
	
	printf("\ntopBorderType\n");
	for (int i = 0; i < width; ++i)
		printf("%4d", borderTypeToPrint[TOP][i]);
	printf("\n");


	printf("\nleftBorderType\n");
	for (int i = 0; i < length; ++i)
		printf("%4d", borderTypeToPrint[LEFT][i]);
	printf("\n");


	printf("\nbottomBorderType\n");
	for (int i = 0; i < width; ++i)
		printf("%4d", borderTypeToPrint[BOTTOM][i]);
	printf("\n");


	printf("\nrightBorderType\n");
	for (int i = 0; i < length; ++i)
		printf("%4d", borderTypeToPrint[RIGHT][i]);
	printf("\n");
	
	
	printf("\ntopBlockBorder\n");
	for (int i = 0; i < width; ++i)
		printf("%6.1f", blockBorderToPrint[TOP][i]);
	printf("\n");


	printf("\nleftBlockBorder\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", blockBorderToPrint[LEFT][i]);
	printf("\n");


	printf("\nbottomBlockBorder\n");
	for (int i = 0; i <width; ++i)
		printf("%6.1f", blockBorderToPrint[BOTTOM][i]);
	printf("\n");


	printf("\nrightBlockBorder\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", blockBorderToPrint[RIGHT][i]);
	printf("\n");
	
	printf("\ntopExternalBorder\n");
	for (int i = 0; i < width; ++i)
		printf("%6.1f", externalBorderToPrint[TOP][i]);
	printf("\n");


	printf("\nleftExternalBorder\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", externalBorderToPrint[LEFT][i]);
	printf("\n");


	printf("\nbottomExternalBorder\n");
	for (int i = 0; i <width; ++i)
		printf("%6.1f", externalBorderToPrint[BOTTOM][i]);
	printf("\n");


	printf("\nrightExternalBorder\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", externalBorderToPrint[RIGHT][i]);
	printf("\n");


	printf("\n\n\n\n\n\n\n");
}