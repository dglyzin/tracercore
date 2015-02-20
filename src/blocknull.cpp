/*
 * BlockNull.cpp
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#include "blocknull.h"

BlockNull::BlockNull(int _length, int _width, int _lengthMove, int _widthMove, int _nodeNumber, int _deviceNumber) : Block( _length, _width, _lengthMove, _widthMove, _nodeNumber, _deviceNumber ) {}

BlockNull::~BlockNull() {
}
