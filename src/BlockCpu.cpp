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





































