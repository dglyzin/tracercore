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

	virtual ~BlockCpu();

	bool isRealBlock() { return true; }

	void prepareData();

	void courted(double dX2, double dY2, double dT);

	int getBlockType() { return CPU; }

	void setPartBorder(int type, int side, int move, int borderLength);

	void print();
	void printMatrix();

	void createBlockBorder(int side, int neighborType);
	void setExternalBorder(int side, double* _externalBorder);
};

#endif /* SRC_BLOCKCPU_H_ */
