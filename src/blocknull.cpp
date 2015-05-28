/*
 * BlockNull.cpp
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#include "blocknull.h"

BlockNull::BlockNull(int _dimension, int _xCount, int _yCount, int _zCount,
		int _xOffset, int _yOffset, int _zOffset,
		int _nodeNumber, int _deviceNumber,
		int _haloSize, int _cellSize) :
				Block( _dimension, _xCount, _yCount, _zCount,
				_xOffset, _yOffset, _zOffset,
				_nodeNumber, _deviceNumber,
				_haloSize, _cellSize) {}

BlockNull::~BlockNull() {
}
