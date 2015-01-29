/*
 * BlockGpu.h
 *
 *  Created on: 29 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKGPU_H_
#define SRC_BLOCKGPU_H_

#include "Block.h"

#define BLOCK_LENGHT_SIZE 32
#define BLOCK_WIDTH_SIZE 16

class BlockGpu: public Block {
public:
	BlockGpu(int _length, int _width, int _lengthMove, int _widthMove, int _world_rank);
	virtual ~BlockGpu();

	// TODO DEvice0,1,2??
	int getBlockType() { return DEVICE0; }

	void courted(double dX2, double dY2, double dT);
};

#endif /* SRC_BLOCKGPU_H_ */
