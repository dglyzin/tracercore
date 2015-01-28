/*
 * BlockCpu.cpp
 *
 *  Created on: 20 янв. 2015 г.
 *      Author: frolov
 */

#include "BlockCpu.h"

using namespace std;

BlockCpu::BlockCpu(int _length, int _width, int _lengthMove, int _widthMove, int _world_rank) : Block(  _length, _width, _lengthMove, _widthMove, _world_rank  ) {

	matrix = new double* [length];

	for(int i = 0; i < length; i++)
		matrix[i] = new double[width];

	for (int i = 0; i < length; ++i)
		for (int j = 0; j < width; ++j)
			matrix[i][j] = 0;

	/*
	 * Типы границ блока. Выделение памяти.
	 */
	/*topBorderType = new int[width];
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
		rightBorderType[i] = BY_FUNCTION;*/
	borderType = new int* [BORDER_COUNT];

	borderType[TOP] = new int[width];
	for(int i = 0; i < width; i++)
		borderType[TOP][i] = BY_FUNCTION;

	borderType[LEFT] = new int[length];
	for (int i = 0; i < length; ++i)
		borderType[LEFT][i] = BY_FUNCTION;

	borderType[BOTTOM] = new int[width];
	for(int i = 0; i < width; i++)
		borderType[BOTTOM][i] = BY_FUNCTION;

	borderType[RIGHT] = new int[length];
	for (int i = 0; i < length; ++i)
		borderType[RIGHT][i] = BY_FUNCTION;

	/*
	 * Границы самого блока.
	 * Это он будет отдавать. Выделение памяти.
	 */
	/*topBlockBorder = new double[width];
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
		rightBlockBorder[i] = 0;*/
	blockBorder = new double* [BORDER_COUNT];

	blockBorder[TOP] = new double[width];
	for(int i = 0; i < width; i++)
		blockBorder[TOP][i] = 0;

	blockBorder[LEFT] = new double[length];
	for (int i = 0; i < length; ++i)
		blockBorder[LEFT][i] = 0;

	blockBorder[BOTTOM] = new double[width];
	for(int i = 0; i < width; i++)
		blockBorder[BOTTOM][i] = 0;

	blockBorder[RIGHT] = new double[length];
	for (int i = 0; i < length; ++i)
		blockBorder[RIGHT][i] = 0;

	/*
	 * Внешние границы блока.
	 * Сюда будет приходить информация.
	 */
	/*topExternalBorder = new double[width];
	for(int i = 0; i < width; i++)
		topExternalBorder[i] = 100;//100 * cos( (i - width/2. ) / (width/2.));

	leftExternalBorder = new double[length];
	for (int i = 0; i < length; ++i)
		leftExternalBorder[i] = 10;

	bottomExternalBorder = new double[width];
	for(int i = 0; i < width; i++)
		bottomExternalBorder[i] = 10;

	rightExternalBorder = new double[length];
	for (int i = 0; i < length; ++i)
		rightExternalBorder[i] = 10;*/
	externalBorder = new double* [BORDER_COUNT];

	externalBorder[TOP] = new double[width];
	for(int i = 0; i < width; i++)
		externalBorder[TOP][i] = 100;//100 * cos( (i - width/2. ) / (width/2.));

	externalBorder[LEFT] = new double[length];
	for (int i = 0; i < length; ++i)
		externalBorder[LEFT][i] = 10;

	externalBorder[BOTTOM] = new double[width];
	for(int i = 0; i < width; i++)
		externalBorder[BOTTOM][i] = 10;

	externalBorder[RIGHT] = new double[length];
	for (int i = 0; i < length; ++i)
		externalBorder[RIGHT][i] = 10;
}

BlockCpu::~BlockCpu() {
	// TODO Auto-generated destructor stub
}


void BlockCpu::prepareData() {
	for (int i = 0; i < width; ++i)
		blockBorder[TOP][i] = matrix[0][i];

	for (int i = 0; i < length; ++i)
		blockBorder[LEFT][i] = matrix[i][0];

	for (int i = 0; i < width; ++i)
		blockBorder[BOTTOM][i] = matrix[length-1][i];

	for (int i = 0; i < length; ++i)
		blockBorder[RIGHT][i] = matrix[i][width-1];
}

void BlockCpu::courted(double dX2, double dY2, double dT) {
	/*
	 * Теплопроводность
	 */

	double** newMatrix = new double* [length];
	for(int i = 0; i < length; i++)
		newMatrix[i] = new double[width];

# pragma omp parallel
{
	double top, left, bottom, right, cur;

# pragma omp for
	for (int i = 0; i < length; ++i)
		for (int j = 0; j < width; ++j) {
			if( i == 0 )
				if( borderType[TOP][j] == BY_FUNCTION ) {
					newMatrix[i][j] = externalBorder[TOP][j];
					continue;
				}
				else
					top = externalBorder[TOP][j];
			else
				top = matrix[i - 1][j];


			if( j == 0 )
				if( borderType[LEFT][i] == BY_FUNCTION ) {
					newMatrix[i][j] = externalBorder[LEFT][i];
					continue;
				}
				else
					left = externalBorder[LEFT][i];
			else
				left = matrix[i][j - 1];


			if( i == length - 1 )
				if( borderType[BOTTOM][j] == BY_FUNCTION ) {
					newMatrix[i][j] = externalBorder[BOTTOM][j];
					continue;
				}
				else
					bottom = externalBorder[BOTTOM][j];
			else
				bottom = matrix[i + 1][j];


			if( j == width - 1 )
				if( borderType[RIGHT][i] == BY_FUNCTION ) {
					newMatrix[i][j] = externalBorder[RIGHT][i];
					continue;
				}
				else
					right = externalBorder[RIGHT][i];
			else
				right = matrix[i][j + 1];


			cur = matrix[i][j];

			newMatrix[i][j] = cur + dT * ( ( left - 2*cur + right )/dX2 + ( top - 2*cur + bottom )/dY2  );
		}
}

	double** tmp = matrix;

	matrix = newMatrix;

	for(int i = 0; i < length; i++)
		delete tmp[i];
	delete tmp;
}

void BlockCpu::print(int locationNode) {
	printf("FROM NODE #%d", locationNode);

	printf("\nLength: %d, Width: %d, World_Rank: %d\n", length, width, nodeNumber);

	printf("\nMatrix:\n");
	for (int i = 0; i < length; ++i)
	{
		for (int j = 0; j < width; ++j)
			printf("%6.1f ", matrix[i][j]);
		printf("\n");
	}


	printf("\ntopBorderType\n");
	for (int i = 0; i < width; ++i)
		printf("%4d", borderType[TOP][i]);
	printf("\n");


	printf("\nleftBorderType\n");
	for (int i = 0; i < length; ++i)
		printf("%4d", borderType[LEFT][i]);
	printf("\n");


	printf("\nbottomBorderType\n");
	for (int i = 0; i < width; ++i)
		printf("%4d", borderType[BOTTOM][i]);
	printf("\n");


	printf("\nrightBorderType\n");
	for (int i = 0; i < length; ++i)
		printf("%4d", borderType[RIGHT][i]);
	printf("\n");


	printf("\ntopBlockBorder\n");
	for (int i = 0; i < width; ++i)
		printf("%6.1f", blockBorder[TOP][i]);
	printf("\n");


	printf("\nleftBlockBorder\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", blockBorder[LEFT][i]);
	printf("\n");


	printf("\nbottomBlockBorder\n");
	for (int i = 0; i <width; ++i)
		printf("%6.1f", blockBorder[BOTTOM][i]);
	printf("\n");


	printf("\nrightBlockBorder\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", blockBorder[RIGHT][i]);
	printf("\n");


	printf("\ntopExternalBorder\n");
	for (int i = 0; i < width; ++i)
		printf("%6.1f", externalBorder[TOP][i]);
	printf("\n");


	printf("\nleftExternalBorder\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", externalBorder[LEFT][i]);
	printf("\n");


	printf("\nbottomExternalBorder\n");
	for (int i = 0; i <width; ++i)
		printf("%6.1f", externalBorder[BOTTOM][i]);
	printf("\n");


	printf("\nrightExternalBorder\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", externalBorder[RIGHT][i]);
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





































