/*
 * BlockCpu.h
 *
 *  Created on: 20 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKCPU_H_
#define SRC_BLOCKCPU_H_

#include "Block.h"

class BlockCpu: public Block {
public:
	BlockCpu(int _length, int _width,
			int* _topBoundaryType, int* _leftBoundaryType, int* _bottomBoundaryType, int* _rightBoundaryType,
			double* _topBlockBoundary, double* _leftBlockBoundary, double* _bottomBlockBoundary, double* _rightBlockBoundary,
			double* _topExternalBoundary, double* _leftExternalBoundary, double* _bottomExternalBoundary, double* _rightExternalBoundary)/* :
				Block(_length, _width,
					_topBoundaryType, _leftBoundaryType, _bottomBoundaryType, _rightBoundaryType,
					_topBlockBoundary, _leftBlockBoundary, _bottomBlockBoundary, _rightBlockBoundary,
					_topExternalBoundary, _leftExternalBoundary, _bottomExternalBoundary, _rightExternalBoundary)*/;


	virtual ~BlockCpu();

	void prepareData();
	bool isRealBlock();

	void courted();

	void print(int locationNode);
	void printMatrix();
};

#endif /* SRC_BLOCKCPU_H_ */
