/*
 * Domain.h
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_DOMAIN_H_
#define SRC_DOMAIN_H_

#include <fstream>
#include <string.h>
#include "blockcpu.h"
#include "blockgpu.h"
#include "blocknull.h"
#include "interconnect.h"

/*
 * Основной управляющий класс приложения.
 * Создает блоки (BlockCpu, BlockGpu, BlockNull) и их соединения (Interconnect).
 */

class Domain {
public:
	Domain(int _world_rank, int _world_size, char* path);
	virtual ~Domain();

	/*
	 * Полный расчет
	 */
	void count();

	/*
	 * Выполнение одной итерации (одного шага)
	 */
	void nextStep(double dX, double dY, double dT);

	void print(char* path);

	/*
	 * Чтение с файла.
	 */
	void readFromFile(char* path);

	/*
	 * Возвращает суммарное количество узлов области.
	 * Сумма со всех блоков.
	 */
	int getCountGridNodes();
	int getRepeatCount();

	int getCountCpuBlocks();
	int getCountGpuBlocks();

	int realBlockCount();

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
	 * Номер потока
	 */
	int world_rank;

	/*
	 * Количество потоков в целом
	 */
	int world_size;

	/*
	 * Количество блоков
	 */
	int blockCount;

	/*
	 * Количество соединений между блоками
	 */
	int connectionCount;

	/*
	 * Размеры области
	 * Вся область - прямоугольник, внутри которого гарантирвано размещаются все блоки.
	 */
	int lengthArea;
	int widthArea;

	MPI_Status status;

	/*
	 * Чтение размеров области
	 */
	void readLengthAndWidthArea(std::ifstream& in);

	/*
	 * Чтение блока
	 */
	Block* readBlock(std::ifstream& in);

	/*
	 * Чтение соединения
	 */
	Interconnect* readConnection(std::ifstream& in);
};

#endif /* SRC_DOMAIN_H_ */