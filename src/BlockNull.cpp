/*
 * BlockNull.cpp
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#include "BlockNull.h"

BlockNull::BlockNull() : Block() {}

BlockNull::BlockNull(int _world_rank) : Block(_world_rank) {}

BlockNull::BlockNull(int _length, int _width, int _world_rank) : Block( _length, _width, _world_rank ) {}

BlockNull::~BlockNull() {
	// TODO Auto-generated destructor stub
}

