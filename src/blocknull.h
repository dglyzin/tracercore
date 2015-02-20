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
	BlockNull(int _length, int _width, int _lengthMove, int _widthMove, int _nodeNumber, int _deviceNumber);
	virtual ~BlockNull();

	bool isRealBlock() { return false; }

	int getBlockType() { return NULL_BLOCK; }
};

#endif /* SRC_BLOCKNULL_H_ */
