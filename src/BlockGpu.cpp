/*
 * BlockGpu.cpp
 *
 *  Created on: 29 янв. 2015 г.
 *      Author: frolov
 */

#include "BlockGpu.h"

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
