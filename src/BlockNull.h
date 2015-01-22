/*
 * BlockNull.h
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKNULL_H_
#define SRC_BLOCKNULL_H_

#include "Block.h"

class BlockNull: public Block {
public:
	BlockNull();
	BlockNull(int _length, int _width);
	virtual ~BlockNull();

	bool isRealBlock() { return false; }
};

#endif /* SRC_BLOCKNULL_H_ */
