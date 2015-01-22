/*
 * BlockCpu.cpp
 *
 *  Created on: 20 янв. 2015 г.
 *      Author: frolov
 */

#include "BlockCpu.h"

using namespace std;

BlockCpu::BlockCpu(int _length, int _width) : Block( _length, _width ) {

	matrix = new double* [length];

	for(int i = 0; i < length; i++)
		matrix[i] = new double[width];

	for (int i = 0; i < length; ++i)
		for (int j = 0; j < width; ++j)
			matrix[i][j] = 0;

	/* Типы границ блока. Выделение памяти */
	topBoundaryType = new int[width];
	for(int i = 0; i < width; i++)
		topBoundaryType[i] = 0;

	leftBoundaryType = new int[length];
	for (int i = 0; i < length; ++i)
		leftBoundaryType[i] = 0;

	bottomBoundaryType = new int[width];
	for(int i = 0; i < width; i++)
		bottomBoundaryType[i] = 0;

	rightBoundaryType = new int[length];
	for (int i = 0; i < length; ++i)
		rightBoundaryType[i] = 0;

	/* Границы самого блока. Это он будет отдавать. Выделение памяти. */
	topBlockBoundary = new double[width];
	for(int i = 0; i < width; i++)
		topBlockBoundary[i] = 0;

	leftBlockBoundary = new double[length];
	for (int i = 0; i < length; ++i)
		leftBlockBoundary[i] = 0;

	bottomBlockBoundary = new double[width];
	for(int i = 0; i < width; i++)
		bottomBlockBoundary[i] = 0;

	rightBlockBoundary = new double[length];
	for (int i = 0; i < length; ++i)
		rightBlockBoundary[i] = 0;

	/* Внешние границы блока. Сюда будет приходить информация. */
	topExternalBoundary = new double[width];
	for(int i = 0; i < width; i++)
		topExternalBoundary[i] = 0;

	leftExternalBoundary = new double[length];
	for (int i = 0; i < length; ++i)
		leftExternalBoundary[i] = 0;

	bottomExternalBoundary = new double[width];
	for(int i = 0; i < width; i++)
		bottomExternalBoundary[i] = 0;

	rightExternalBoundary = new double[length];
	for (int i = 0; i < length; ++i)
		rightExternalBoundary[i] = 0;

}

BlockCpu::~BlockCpu() {
	// TODO Auto-generated destructor stub
}


void BlockCpu::prepareData() {
	for (int i = 0; i < width; ++i)
		topBlockBoundary[i] = matrix[0][i];

	for (int i = 0; i < length; ++i)
		leftBlockBoundary[i] = matrix[i][0];

	for (int i = 0; i < width; ++i)
		bottomBlockBoundary[i] = matrix[length-1][i];

	for (int i = 0; i < length; ++i)
		rightBlockBoundary[i] = matrix[i][width-1];
}

bool BlockCpu::isRealBlock() {
	return true;
}

void BlockCpu::courted() {
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
				if( topBoundaryType[j] == 1 ) {
					newMatrix[i][j] = topExternalBoundary[j];
					continue;
				}
				else
					top = topExternalBoundary[j];
			else
				top = matrix[i - 1][j];


			if( i == length - 1 )
				if( bottomBoundaryType[j] == 1 ) {
					newMatrix[i][j] = bottomExternalBoundary[j];
					continue;
				}
				else
					bottom = bottomExternalBoundary[j];
			else
				bottom = matrix[i + 1][j];


			if( j == 0 )
				if( leftBoundaryType[i] == 1 ) {
					newMatrix[i][j] = leftExternalBoundary[i];
					continue;
				}
				else
					left = leftExternalBoundary[i];
			else
				left = matrix[i][j - 1];


			if( j == width - 1 )
				if( rightBoundaryType[i] == 1 ) {
					newMatrix[i][j] = rightExternalBoundary[i];
					continue;
				}
				else
					right = rightExternalBoundary[i];
			else
				right = matrix[i][j + 1];


			cur = matrix[i][j];

			newMatrix[i][j] = cur + dT * ( ( left - 2*cur + right )/dX2 + ( top - 2*cur + bottom )/dY2  );
		}

	/*for (int i = 0; i < length; ++i)
		for (int j = 0; j < width; ++j)
			matrix[i][j] = newMatrix[i][j];*/

	double** tmp = matrix;

	matrix = newMatrix;

	for(int i = 0; i < length; i++)
			delete tmp[i];
	delete tmp;
}

void BlockCpu::print(int locationNode) {
	printf("FROM NODE #%d", locationNode);
	printf("\nMatrix:\n");
	for (int i = 0; i < length; ++i)
	{
		for (int j = 0; j < width; ++j)
			printf("%6.1f ", matrix[i][j]);
		printf("\n");
	}


	printf("\ntopBoundaryType\n");
	for (int i = 0; i < width; ++i)
		printf("%4d", topBoundaryType[i]);
	printf("\n");


	printf("\nleftBoundaryType\n");
	for (int i = 0; i < length; ++i)
		printf("%4d", leftBoundaryType[i]);
	printf("\n");


	printf("\nbottomBoundaryType\n");
	for (int i = 0; i < width; ++i)
		printf("%4d", bottomBoundaryType[i]);
	printf("\n");


	printf("\nrightBoundaryType\n");
	for (int i = 0; i < length; ++i)
		printf("%4d", rightBoundaryType[i]);
	printf("\n");


	printf("\ntopBlockBoundary\n");
	for (int i = 0; i < width; ++i)
		printf("%6.1f", topBlockBoundary[i]);
	printf("\n");


	printf("\nleftBlockBoundary\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", leftBlockBoundary[i]);
	printf("\n");


	printf("\nbottomBlockBoundary\n");
	for (int i = 0; i <width; ++i)
		printf("%6.1f", bottomBlockBoundary[i]);
	printf("\n");


	printf("\nrightBlockBoundary\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", rightBlockBoundary[i]);
	printf("\n");


	printf("\ntopExternalBoundary\n");
	for (int i = 0; i < width; ++i)
		printf("%6.1f", topExternalBoundary[i]);
	printf("\n");


	printf("\nleftExternalBoundary\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", leftExternalBoundary[i]);
	printf("\n");


	printf("\nbottomExternalBoundary\n");
	for (int i = 0; i <width; ++i)
		printf("%6.1f", bottomExternalBoundary[i]);
	printf("\n");


	printf("\nrightExternalBoundary\n");
	for (int i = 0; i < length; ++i)
		printf("%6.1f", rightExternalBoundary[i]);
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





































