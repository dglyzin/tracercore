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
	BlockCpu(int _length, int _width, int _lengthMove, int _widthMove, int _world_rank);

	virtual ~BlockCpu();

	void prepareData();
	bool isRealBlock() { return true; }

	void courted();

	int getBlockType() { return CPU; }

	void print(int locationNode);
	void printMatrix();

	void setTopExternalBorder(double* _topExternalBorder) {
		if(topExternalBorder) delete topExternalBorder;
		topExternalBorder = _topExternalBorder;
	}
};

#endif /* SRC_BLOCKCPU_H_ */
