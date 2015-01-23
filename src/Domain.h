/*
 * Domain.h
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_DOMAIN_H_
#define SRC_DOMAIN_H_

#include "Interconnect.h"
#include "BlockCpu.h"
#include "BlockNull.h"

#include <cmath>

#define b0_length 50
#define b0_width 25
#define b0_moveL 35
#define b0_moveW 0

#define b1_length 25
#define b1_width 50
#define b1_moveL 50
#define b1_moveW 25

#define b2_length 50
#define b2_width 25
#define b2_moveL 0
#define b2_moveW 38

#define b3_length 50
#define b3_width 25
#define b3_moveL 35
#define b3_moveW 75


#define b0_right_border_move 15

#define b1_top_border_move 13
#define b1_left_border_move 0
#define b1_right_border_move 0

#define b2_bottom_border_move 0

#define b3_left_border_move 15


#define b0_b1_border_length 25
#define b1_b2_border_length 25
#define b1_b3_border_length 25

/*
 * Основной управляющий класс приложения.
 * Создает блоки (BlockCpu, BlockGpu, BlockNull) и их соединения (Interconnect).
 */

class Domain {
public:
	Domain(int world_rank, int world_size, int blockCount, int borderCount);
	virtual ~Domain();

	void calc(int world_rank, int blockCount, int borderCount);

	void print(int world_rank, int blockCount);

private:
	/*
	 * Массив блоков.
	 * Массив указателей на блоки.
	 * Содержит все блоки потока (реальные и нереальные).
	 * Каждый поток исполнения содержит одинаковое количество блоков.
	 */
	Block** mBlocks;

	/*
	 * Массив соединений.
	 * Массив указателей на соединения.
	 * Содержит все соединения.
	 * Каждый поток исполнения содержит одинаковое количество соединений.
	 * Каждый поток исполнеия вызывает пересылку на каждом из них.
	 * Реальная пересылка произойдет только если вызов пришел с коррекного потока исполения.
	 */
	Interconnect** mInterconnects;

	/*
	 * Массивы данных о блоках.
	 * Служебные переменные.
	 * TODO
	 * Сделать подгрузку с файлов
	 */
	int blockLengthSize[4];
	int blockWidthSize[4];

	int blockMoveLenght[4];
	int blockMoveWidth[4];

	MPI_Status status;

	void setDefaultValue();
};

#endif /* SRC_DOMAIN_H_ */
