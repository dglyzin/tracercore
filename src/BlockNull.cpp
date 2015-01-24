/*
 * BlockNull.cpp
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#include "BlockNull.h"

BlockNull::BlockNull() : Block() {}

BlockNull::BlockNull(int _length, int _width, int _lengthMove, int _widthMove, int _world_rank) : Block( _length, _width, _lengthMove, _widthMove, _world_rank ) {}

BlockNull::~BlockNull() {
	// TODO Auto-generated destructor stub
}

