/*
 * BlockCpu.cpp
 *
 *  Created on: 20 янв. 2015 г.
 *      Author: frolov
 */

#include "BlockCpu.h"

using namespace std;

BlockCpu::BlockCpu(int _length, int _width, int _world_rank) : Block( _length, _width, _world_rank ) {

	matrix = new double* [length];

	for(int i = 0; i < length; i++)
		matrix[i] = new double[width];

	for (int i = 0; i < length; ++i)
		for (int j = 0; j < width; ++j)
			matrix[i][j] = 0;

	/*
	 * Типы границ блока. Выделение памяти.
	 */
	topBorderType = new int[width];
	for(int i = 0; i < width; i++)
		topBorderType[i] = BY_FUNCTION;

	leftBorderType = new int[length];
	for (int i = 0; i < length; ++i)
		leftBorderType[i] = BY_FUNCTION;

	bottomBorderType = new int[width];
	for(int i = 0; i < width; i++)
		bottomBorderType[i] = BY_FUNCTION;

	rightBorderType = new int[length];
	for (int i = 0; i < length; ++i)
		rightBorderType[i] = BY_FUNCTION;

	/*
	 * Границы самого блока.
	 * Это он будет отдавать. Выделение памяти.
	 */
	topBlockBorder = new double[width];
	for(int i = 0; i < width; i++)
		topBlockBorder[i] = 0;

	leftBlockBorder = new double[length];
	for (int i = 0; i < length; ++i)
		leftBlockBorder[i] = 0;

	bottomBlockBorder = new double[width];
	for(int i = 0; i < width; i++)
		bottomBlockBorder[i] = 0;

	rightBlockBorder = new double[length];
	for (int i = 0; i < length; ++i)
		rightBlockBorder[i] = 0;

	/*
	 * Внешние границы блока.
	 * Сюда будет приходить информация.
	 */
	topExternalBorder = new double[width];
	for(int i = 0; i < width; i++)
		topExternalBorder[i] = 0;

	leftExternalBorder = new double[length];
	for (int i = 0; i < length; ++i)
		leftExternalBorder[i] = 0;

	bottomExternalBorder = new double[width];
	for(int i = 0; i < width; i++)
		bottomExternalBorder[i] = 0;

	rightExternalBorder = new double[length];
	for (int i = 0; i < length; ++i)
		rightExternalBorder[i] = 0;
}

BlockCpu::~BlockCpu() {
	// TODO Auto-generated destructor stub
}


void BlockCpu::prepareData() {
	for (int i = 0; i < width; ++i)
		topBlockBorder[i] = matrix[0][i];

	for (int i = 0; i < length; ++i)
		leftBlockBorder[i] = matrix[i][0];

	for (int i = 0; i < width; ++i)
		bottomBlockBorder[i] = matrix[length-1][i];

	for (int i = 0; i < length; ++i)
		rightBlockBorder[i] = matrix[i][width-1];
}

void BlockCpu::courted() {
	/*
	 * Теплопроводность
	 */
	double dX = 0.5/width;
	double dY = 1./length;

	double dX2 = dX*dX;
	double dY2 = dY*dY;

	double dT = ( dX2 * dY2 ) / ( 2 * ( dX2 + dY2 ) );

	double top, left, bottom, right, cur;

	double** newMatrix = new double* [length];
	for(int i = 0; i < length; i++)
		newMatrix[i] = new double[width];


	for (int i = 0; i < length; ++i)
		for (int j = 0; j < width; ++j) {
			if( i == 0 )
				if( topBorderType[j] == BY_FUNCTION ) {
					newMatrix[i][j] = topExternalBorder[j];
					continue;
				}
				else
					top = topExternalBorder[j];
			else
				top = matrix[i - 1][j];


			if( i == length - 1 )
				if( bottomBorderType[j] == BY_FUNCTION ) {
					newMatrix[i][j] = bottomExternalBorder[j];
					continue;
				}
				else
					bottom = bottomExternalBorder[j];
			else
				bottom = matrix[i + 1][j];


			if( j == 0 )
				if( leftBorderType[i] == BY_FUNCTION ) {
					newMatrix[i][j] = leftExternalBorder[i];
					continue;
				}
				else
					left = leftExternalBorder[i];
			else
				left = matrix[i][j - 1];


			if( j == width - 1 )
				if( rightBorderType[i] == BY_FUNCTION ) {
					newMatrix[i][j] = rightExternalBorder[i];
					continue;
				}
				else
					right = rightExternalBorder[i];
			else
				right = matrix[i][j + 1];


			cur = matrix[i][j];

			newMatrix[i][j] = cur + dT * ( ( left - 2*cur + right )/dX2 + ( top - 2*cur + bottom )/dY2  );
		}

	double** tmp = matrix;

	matrix = newMatrix;

	for(int i = 0; i < length; i++)
		delete tmp[i];
	delete tmp;
}

void BlockCpu::print(int locationNode) {
	printf("FROM NODE #%d", locationNode);

	printf("\nLength: %d, Width: %d, World_Rank: %d\n", length, width, world_rank);

	printf("\nMatrix:\n");
	for (int i = 0; i < length; ++i)
	{
		for (int j = 0; j < width; ++j)
			printf("%6.1f ", matrix[i][j]);
		printf("\n");
	}


	printf("\ntopBorderType\n");
	for (int i = 0; i < width; ++i)
		printf("%4d", topBorderType[i]);
	printf("\n");


	printf("\nleftBorderType\n");
	for (int i = 0; i < length; ++i)
		printf("%4d", leftBorderType[i]);
	printf("\n");


	printf("\nbottomBorderType\n");
	for (int i = 0; i < width; ++i)
		printf("%4d", bottomBorderType[i]);
	printf("\n");


	printf("\nrightBorderType\n");
	for (int i = 0; i < length; ++i)
		printf("%4d", rightBorderType[i]);
	printf("\n");


	printf("\ntopBlockBorder\n");
	for (int i = 0; i < width; ++i)
		printf("%6.1f", topBlockBorder[i]);
	printf("\n");


	printf("\nleftBlockBorder\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", leftBlockBorder[i]);
	printf("\n");


	printf("\nbottomBlockBorder\n");
	for (int i = 0; i <width; ++i)
		printf("%6.1f", bottomBlockBorder[i]);
	printf("\n");


	printf("\nrightBlockBorder\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", rightBlockBorder[i]);
	printf("\n");


	printf("\ntopExternalBorder\n");
	for (int i = 0; i < width; ++i)
		printf("%6.1f", topExternalBorder[i]);
	printf("\n");


	printf("\nleftExternalBorder\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", leftExternalBorder[i]);
	printf("\n");


	printf("\nbottomExternalBorder\n");
	for (int i = 0; i <width; ++i)
		printf("%6.1f", bottomExternalBorder[i]);
	printf("\n");


	printf("\nrightExternalBorder\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", rightExternalBorder[i]);
	printf("\n");


	printf("\n\n\n\n\n\n\n");
}

void BlockCpu::printMatrix() {
	printf("\nMatrix:\n");
	for (int i = 0; i < length; ++i)
	{
		for (int j = 0; j < width; ++j)
			printf("%6.1f ", matrix[i][j]);
		printf("\n");
	}
}





































