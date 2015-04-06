/*
 * BlockNull.h
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKNULL_H_
#define SRC_BLOCKNULL_H_

#include "block.h"

/*
 * Блок - загушка.
 * Отвечает false на вопрос о своей реальности.
 * Остальные функции своей предка не переопределяет.
 */

class BlockNull: public Block {
public:
	BlockNull(int _dimension, int _xCount, int _yCount, int _zCount,
				int _xOffset, int _yOffset, int _zOffset,
				int _nodeNumber, int _deviceNumber,
				int _haloSize, int _cellSize);
	virtual ~BlockNull();

	bool isRealBlock() { return false; }

	int getBlockType() { return NULL_BLOCK; }
};

#endif /* SRC_BLOCKNULL_H_ */
