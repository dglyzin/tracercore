/*
 * BlockGpu.cpp
 *
 *  Created on: 29 янв. 2015 г.
 *      Author: frolov
 */

#include "BlockGpu.h"

/*
 * Функция ядра.
 * Расчет теплоемкости на видеокарте.
 * Логика функции аналогична функции для центрального процессора.
 */
__global__ void calc ( double** matrix, double** newMatrix, int length, int width, double dX2, double dY2, double dT, int **borderType, double** externalBorder ) {

	double top, left, bottom, right, cur;

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int i = BLOCK_LENGHT_SIZE * bx + tx;
	int j = BLOCK_WIDTH_SIZE * by + ty;

	if( i > length || j > width )
		return;

	if( i == 0 )
		if( borderType[TOP][j] == BY_FUNCTION ) {
			newMatrix[i][j] = externalBorder[TOP][j];
			return;
		}
		else
			top = externalBorder[TOP][j];
	else
		top = matrix[i - 1][j];


	if( j == 0 )
		if( borderType[LEFT][i] == BY_FUNCTION ) {
			newMatrix[i][j] = externalBorder[LEFT][i];
			return;
		}
		else
			left = externalBorder[LEFT][i];
	else
		left = matrix[i][j - 1];


	if( i == length - 1 )
		if( borderType[BOTTOM][j] == BY_FUNCTION ) {
			newMatrix[i][j] = externalBorder[BOTTOM][j];
			return;
		}
		else
			bottom = externalBorder[BOTTOM][j];
	else
		bottom = matrix[i + 1][j];


	if( j == width - 1 )
		if( borderType[RIGHT][i] == BY_FUNCTION ) {
			newMatrix[i][j] = externalBorder[RIGHT][i];
			return;
		}
		else
			right = externalBorder[RIGHT][i];
	else
		right = matrix[i][j + 1];


	cur = matrix[i][j];

	newMatrix[i][j] = cur + dT * ( ( left - 2*cur + right )/dX2 + ( top - 2*cur + bottom )/dY2  );
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
__global__ void copyBorderFromMatrix ( double* dest, double** matrix, int side, int borderLength )
{
	int bx  = blockIdx.x;
	int tx  = threadIdx.x;
	int idx  = BLOCK_SIZE * bx + tx;
	
	if(idx < borderLength)
		switch (side) {
			case TOP:
				dest[idx] = matrix[0][idx];
				break;
			case LEFT:
				dest[idx] = matrix[idx][0];
				break;
			case BOTTOM:
				dest[idx] = matrix[borderLength - 1][idx];
				break;
			case RIGHT:
				dest[idx] = matrix[idx][borderLength - 1];
				break;
			default:
				break;
		}
}

BlockGpu::BlockGpu(int _length, int _width, int _lengthMove, int _widthMove, int _world_rank, int _deviceNumber) : Block(  _length, _width, _lengthMove, _widthMove, _world_rank ) {
	deviceNumber = _deviceNumber;
	
	dim3 threads ( BLOCK_SIZE );
	dim3 blocksLength  ( (int)ceil((double)length / threads.x) );
	dim3 blocksWidth  ( (int)ceil((double)width / threads.x) );
	
	matrix = new double* [length];
	for (int i = 0; i < length; ++i) {
		cudaMalloc ( (void**)&matrix[i], width * sizeof(double) );
		assignDoubleArray <<< blocksWidth, threads >>> ( matrix[i], 0, width );
	}

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
}

BlockGpu::~BlockGpu() {
	// TODO Auto-generated destructor stub
}

void BlockGpu::courted(double dX2, double dY2, double dT) {
	double** newMatrix = new double* [length];
	for (int i = 0; i < length; ++i)
		cudaMalloc ( (void**)&newMatrix[i], width * sizeof(double) );

	dim3 threads ( BLOCK_LENGHT_SIZE, BLOCK_WIDTH_SIZE );
	dim3 blocks  ( (int)ceil((double)length / threads.x), (int)ceil((double)width / threads.y) );

	calc <<< blocks, threads >>> ( matrix, newMatrix, length, width, dX2, dY2, dT, borderType, externalBorder );

	double** tmp = matrix;

	matrix = newMatrix;

	for(int i = 0; i < length; i++)
		cudaFree(tmp[i]);
	delete tmp;
}

void BlockGpu::setPartBorder(int type, int side, int move, int borderLength) {
	if( checkValue(side, move + borderLength) ) {
		printf("\nCritical error!\n");
		exit(1);
	}
	dim3 threads ( BLOCK_SIZE );
	dim3 blocks  ( (int)ceil((double)borderLength / threads.x) );
	
	assignIntArray <<< blocks, threads >>> ( borderType[side] + move, type, borderLength );
}

void BlockGpu::prepareData() {
	dim3 threads ( BLOCK_SIZE );
	dim3 blocksLength  ( (int)ceil((double)length / threads.x) );
	dim3 blocksWidth  ( (int)ceil((double)width / threads.x) );
	
	copyBorderFromMatrix <<< blocksWidth, threads >>> (blockBorder[TOP], matrix, TOP, width);
	copyBorderFromMatrix <<< blocksLength, threads >>> (blockBorder[LEFT], matrix, LEFT, length);
	copyBorderFromMatrix <<< blocksWidth, threads >>> (blockBorder[BOTTOM], matrix, BOTTOM, width);
	copyBorderFromMatrix <<< blocksLength, threads >>> (blockBorder[RIGHT], matrix, RIGHT, length);
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