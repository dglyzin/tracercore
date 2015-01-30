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

#define BLOCK_SIZE 512

class BlockGpu: public Block {
public:
	BlockGpu(int _length, int _width, int _lengthMove, int _widthMove, int _world_rank);
	virtual ~BlockGpu();

	// TODO DEvice0,1,2??
	int getBlockType() { return DEVICE0; }

	void courted(double dX2, double dY2, double dT);

	void setPartBorder(int type, int side, int move, int borderLength);
};

#endif /* SRC_BLOCKGPU_H_ */
