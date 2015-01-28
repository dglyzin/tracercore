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

#include <fstream>
#include <string.h>

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

#define areaLength 85
#define areaWidth 100

/*
 * Основной управляющий класс приложения.
 * Создает блоки (BlockCpu, BlockGpu, BlockNull) и их соединения (Interconnect).
 */

class Domain {
public:
	Domain(int _world_rank, int _world_size, char* path);
	virtual ~Domain();

	void count();
	void nextStep(double dX, double dY, double dT);

	void print(char* path);

	void readFromFile(char* path);

	int getCountGridNodes();

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

	int world_rank;
	int world_size;
	int blockCount;
	int connectionCount;

	int lengthArea;
	int widthArea;

	MPI_Status status;

	void readLengthAndWidthArea(std::ifstream& in);
	Block* readBlock(std::ifstream& in);
	Interconnect* readConnection(std::ifstream& in);

};

#endif /* SRC_DOMAIN_H_ */
