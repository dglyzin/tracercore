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
	Domain(int _world_rank, int _world_size, char* inputFile, int _flags, int _stepCount, double _stopTime, char* loadFile);

	virtual ~Domain();

	/*
	 * Полный расчет
	 */
	void count(char* saveFile);

	/*
	 * Выполнение одной итерации (одного шага)
	 */
	void nextStep(double dX, double dY, double dT);

	/*
	 * Сбор и запись данных в файл.
	 */
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

	/*
	 * Возвращает количество необходимых итераций.
	 */
	int getRepeatCount();

	/*
	 * Количество реальных блоков типа "центральный процессор"
	 */
	int getCountCpuBlocks();

	/*
	 * Количество реальных блоков типа "видеокарта"
	 */
	int getCountGpuBlocks();

	/*
	 * Количество реальных блоков любого типа.
	 */
	int realBlockCount();

	void saveStateToFile(char* path);
	void loadStateFromFile(char* blockLocation, char* dataFile);

	void printStatisticsInfo(char* inputFile, char* outputFile, double calcTime);

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

	int flags;
	int stepCount;
	double stepTime;
	double stopTime;
	double currentTime;

	int repeatCount;

	MPI_Status status;

	/*
	 * Чтение размеров области
	 */
	void readLengthAndWidthArea(std::ifstream& in);

	void readTimeSetting(std::ifstream& in);

	/*
	 * Чтение блока
	 */
	Block* readBlock(std::ifstream& in);

	/*
	 * Чтение соединения
	 */
	Interconnect* readConnection(std::ifstream& in);

	double** collectDataFromNode();

	void prepareData();
	void computeOneStep(double dX2, double dY2, double dT);
	void computeOneStepBorder(double dX2, double dY2, double dT);
	void computeOneStepCenter(double dX2, double dY2, double dT);
	void swapBlockMatrix();
};

#endif /* SRC_DOMAIN_H_ */
