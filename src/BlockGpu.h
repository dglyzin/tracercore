/*
 * BlockGpu.h
 *
 *  Created on: 29 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKGPU_H_
#define SRC_BLOCKGPU_H_

#include "Block.h"

class BlockGpu: public Block {
public:
	BlockGpu(int _length, int _width, int _lengthMove, int _widthMove, int _world_rank);
	virtual ~BlockGpu();
};

#endif /* SRC_BLOCKGPU_H_ */
