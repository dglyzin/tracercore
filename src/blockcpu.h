/*
 * BlockCpu.h
 *
 *  Created on: 20 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKCPU_H_
#define SRC_BLOCKCPU_H_

#include "block.h"

/*
 * Блок работы с данными на центральном процссоре.
 */

class BlockCpu: public Block {
public:
	BlockCpu(int _length, int _width, int _lengthMove, int _widthMove, int _world_rank);

	~BlockCpu();

	bool isRealBlock() { return true; }

	void prepareData();

	void computeOneStep(double dX2, double dY2, double dT);

	int getBlockType() { return CPU; }

	void print();

	double* addNewBlockBorder(int nodeNeighbor, int typeNeighbor, int side, int move, int borderLength);
	double* addNewExternalBorder(int nodeNeighbor, int side, int move, int borderLength, double* border);

	void moveTempBorderVectorToBorderArray();
};

#endif /* SRC_BLOCKCPU_H_ */
