/*
 * BlockGpu.cpp
 *
 *  Created on: 29 янв. 2015 г.
 *      Author: frolov
 */

#include "BlockGpu.h"

__global__ void calc ( double** matrix, double** newMatrix, int length, int width, double dX2, double dY2, double dT, int **borderType, double** externalBorder ) {

	double top, left, bottom, right, cur;

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int j = BLOCK_WIDTH_SIZE * by + ty;
	int i = BLOCK_LENGHT_SIZE * bx + tx;

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

BlockGpu::BlockGpu(int _length, int _width, int _lengthMove, int _widthMove, int _world_rank) : Block(  _length, _width, _lengthMove, _widthMove, _world_rank  ) {
	cudaMalloc ( (void**)&matrix, length * sizeof(double*) );
	
	for (int i = 0; i < length; ++i)
		cudaMalloc ( (void**)&matrix[i], width * sizeof(double) );

	/*
	 * Типы границ блока. Выделение памяти.
	 */
	cudaMalloc ( (void**)&borderType, BORDER_COUNT * sizeof(int*) );

	cudaMalloc ( (void**)&borderType[TOP], width * sizeof(int) );
	for(int i = 0; i < width; i++)
		borderType[TOP][i] = BY_FUNCTION;

	cudaMalloc ( (void**)&borderType[LEFT], length * sizeof(int) );
	for (int i = 0; i < length; ++i)
		borderType[LEFT][i] = BY_FUNCTION;

	cudaMalloc ( (void**)&borderType[BOTTOM], width * sizeof(int) );
	for(int i = 0; i < width; i++)
		borderType[BOTTOM][i] = BY_FUNCTION;

	cudaMalloc ( (void**)&borderType[RIGHT], length * sizeof(int) );
	for (int i = 0; i < length; ++i)
		borderType[RIGHT][i] = BY_FUNCTION;


	/*
	 * Границы самого блока.
	 * Это он будет отдавать. Выделение памяти.
	 */
	cudaMalloc ( (void**)&blockBorder, BORDER_COUNT * sizeof(double*) );

	cudaMalloc ( (void**)&blockBorder[TOP], width * sizeof(double) );
	for(int i = 0; i < width; i++)
		blockBorder[TOP][i] = 0;

	cudaMalloc ( (void**)&blockBorder[LEFT], length * sizeof(double) );
	for (int i = 0; i < length; ++i)
		blockBorder[LEFT][i] = 0;

	cudaMalloc ( (void**)&blockBorder[BOTTOM], width * sizeof(double) );
	for(int i = 0; i < width; i++)
		blockBorder[BOTTOM][i] = 0;

	cudaMalloc ( (void**)&blockBorder[RIGHT], length * sizeof(double) );
	for (int i = 0; i < length; ++i)
		blockBorder[RIGHT][i] = 0;


	/*
	 * Внешние границы блока.
	 * Сюда будет приходить информация.
	 */
	cudaMalloc ( (void**)&externalBorder, BORDER_COUNT * sizeof(double*) );

	cudaMalloc ( (void**)&externalBorder[TOP], width * sizeof(double) );
	for(int i = 0; i < width; i++)
		externalBorder[TOP][i] = 100;//100 * cos( (i - width/2. ) / (width/2.));;

	cudaMalloc ( (void**)&externalBorder[LEFT], length * sizeof(double) );
	for (int i = 0; i < length; ++i)
		externalBorder[LEFT][i] = 10;

	cudaMalloc ( (void**)&externalBorder[BOTTOM], width * sizeof(double) );
	for(int i = 0; i < width; i++)
		externalBorder[BOTTOM][i] = 10;

	cudaMalloc ( (void**)&externalBorder[RIGHT], length * sizeof(double) );
	for (int i = 0; i < length; ++i)
		externalBorder[RIGHT][i] = 10;
}

BlockGpu::~BlockGpu() {
	// TODO Auto-generated destructor stub
}

void BlockGpu::courted(double dX2, double dY2, double dT) {
	double** newMatrix = new double* [length];
	for(int i = 0; i < length; i++)
		newMatrix[i] = new double[width];

	dim3 threads ( BLOCK_LENGHT_SIZE, BLOCK_WIDTH_SIZE );
	dim3 blocks  ( (int)ceil((double)length / threads.x), (int)ceil((double)width / threads.y) );

	calc <<< blocks, threads >>> ( matrix, newMatrix, length, width, dX2, dY2, dT, borderType, externalBorder );

	double** tmp = matrix;

	matrix = newMatrix;

	for(int i = 0; i < length; i++)
		delete tmp[i];
	delete tmp;
}
