/*
 * BlockCpu.h
 *
 *  Created on: 20 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKCPU_H_
#define SRC_BLOCKCPU_H_

#include "Block.h"

/*
 * Блок работы с данными на центральном процссоре.
 */

class BlockCpu: public Block {
public:
	BlockCpu(int _length, int _width);

	virtual ~BlockCpu();

	void prepareData();
	bool isRealBlock() { return true; }

	void courted();

	void print(int locationNode);
	void printMatrix();
};

#endif /* SRC_BLOCKCPU_H_ */
