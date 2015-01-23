/*
 * BlockNull.h
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKNULL_H_
#define SRC_BLOCKNULL_H_

#include "Block.h"

/*
 * Блок - загушка.
 * Отвечает false на вопрос о своей реальности.
 * Остальные функции своей предка не переопределяет.
 */

class BlockNull: public Block {
public:
	BlockNull();
	BlockNull(int _world_rank);
	BlockNull(int _length, int _width, int _world_rank);
	virtual ~BlockNull();

	bool isRealBlock() { return false; }
};

#endif /* SRC_BLOCKNULL_H_ */
