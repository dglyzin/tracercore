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
#include "solvers/solver.h"

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
	void compute(char* saveFile);

	/*
	 * Выполнение одной итерации (одного шага)
	 */
	void nextStep();

	/*
	 * Сбор и запись данных в файл.
	 */
	void print(char* path);
	void printAreaToConsole();
	void printBlocksToConsole();

	/*
	 * Чтение из файла.
	 */
	void readFromFile(char* path);

	/*
	 * Возвращает суммарное количество узлов области.
	 * Сумма со всех блоков.
	 */
	int getGridNodeCount();

	/*
	 * Возвращает количество необходимых итераций.
	 */
	int getRepeatCount();

	/*
	 * Количество реальных блоков типа "центральный процессор"
	 */
	int getCpuBlockCount();

	/*
	 * Количество реальных блоков типа "видеокарта"
	 */
	int getGpuBlockCount();

	/*
	 * Количество реальных блоков любого типа.
	 */
	int realBlockCount();

	void saveStateToFile(char* path);
	void loadStateFromFile(char* blockLocation, char* dataFile);

	void printStatisticsInfo(char* inputFile, char* outputFile, double calcTime, char* statisticsFile);

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
	 * Номер класса солверов
	 * В рамках каждого класса реализуется солвер для цпу и солвер для гпу
	 */
	int mSolverIndex;

	/*
	 * Количество стадий (вычислений правых частей)
	 * используемого солвера
	 */
	int mSolverStageCount;

	/*
	 * Номер потока
	 */
	int mWorldRank;

	/*
	 * Количество потоков в целом
	 */
	int mWorldSize;

	/*
	 * Количество блоков
	 */
	int mBlockCount;

	/*
	 * Количество соединений между блоками
	 */
	int mConnectionCount;

	/*
	 * Размеры области
	 * Вся область - прямоугольник, внутри которого гарантирвано размещаются все блоки.
	 */
	int lengthArea;
	int widthArea;

	int flags;

	int mStepCount;

	double startTime;
	double stopTime;

	double timeStep;
	int mAcceptedStepCount;
	int mRejectedStepCount;


	double currentTime;

	double saveInterval;

	double mDx, mDy, mDz;

	int mCellSize;
	int mHaloSize;

	int mRepeatCount;

	MPI_Status status;

	void readFileStat(std::ifstream& in);
	void readTimeSetting(std::ifstream& in);
	void readSaveInterval(std::ifstream& in);
	void readGridSteps(std::ifstream& in);
	void readCellAndHaloSize(std::ifstream& in);
	void readSolverIndex(std::ifstream& in);
	void readBlockCount(std::ifstream& in);
	void readConnectionCount(std::ifstream& in);

	/*
	 * Чтение блока
	 */
	Block* readBlock(std::ifstream& in);

	/*
	 * Чтение соединения
	 */
	Interconnect* readConnection(std::ifstream& in);

	double** collectDataFromNode();
	double* getBlockCurrentState(int number);

	void prepareData(int stage);
	void prepareDeviceData(int deviceType, int deviceNumber, int stage);
	void processDeviceBlocksBorder(int deviceType, int deviceNumber, int stage);
	void processDeviceBlocksCenter(int deviceType, int deviceNumber, int stage);
	void computeOneStepBorder(int stage);
	void computeOneStepCenter(int stage);
	/*
	 * after every step (successful or not) we update timestep according to an error
	 */
	int checkErrorAndUpdateTimeStep();

	void swapBlockMatrix();
};

#endif /* SRC_DOMAIN_H_ */
