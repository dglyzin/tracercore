/*
 * nullblock.cpp
 *
 *  Created on: 05 нояб. 2015 г.
 *      Author: frolov
 */

#include "nullblock.h"

NullBlock::NullBlock(int _nodeNumber, int _dimension, int _xCount, int _yCount, int _zCount, int _xOffset, int _yOffset, int _zOffset, int _cellSize, int _haloSize) :
		Block(_nodeNumber, _dimension, _xCount, _yCount, _zCount, _xOffset, _yOffset, _zOffset, _cellSize, _haloSize){
	// TODO Auto-generated constructor stub

}

NullBlock::~NullBlock() {
	// TODO Auto-generated destructor stub
}

