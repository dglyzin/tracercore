/*
 * Domain.h
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_DOMAIN_H_
#define SRC_DOMAIN_H_

#include "../blocks/realblock.h"
#include "../blocks/nullblock.h"

#include "../interconnect/transferinterconnect/transferinterconnectsend.h"
#include "../interconnect/transferinterconnect/transferinterconnectrecv.h"
#include "../interconnect/nontransferinterconnect.h"

#include "../processingunit/cpu/cpu1d.h"
#include "../processingunit/cpu/cpu2d.h"
#include "../processingunit/cpu/cpu3d.h"

#include "../processingunit/gpu/gpu1d.h"
#include "../processingunit/gpu/gpu2d.h"
#include "../processingunit/gpu/gpu3d.h"

#include "../utils.h"

/*
 * Основной управляющий класс приложения.
 */

class Domain {
public:
	Domain(int _world_rank, int _world_size, char* inputFile);

	virtual ~Domain();

	/*
	 * Полный расчет
	 */
	void compute(char* inputFile);

	/*
	 * Выполнение одной итерации (одного шага)
	 */
	void nextStep();

	/*
	 * Сбор и запись данных в файл.
	 */
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
	 * Возвращает суммарное количество элементов области.
	 * Сумма со всех блоков.
	 */
	int getGridElementCount();
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

	void saveState(char* inputFile);
	void saveStateToFile(char* path);

	void printStatisticsInfo(char* inputFile, char* outputFile, double calcTime, char* statisticsFile);

	bool isNan();

	void checkOptions(int flags, double _stopTime, char* saveFile);

private:
	ProcessingUnit* cpu;
	ProcessingUnit** gpu;

	int mGpuCount;
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
	 * Структура данных, возвращающая основные параметры солвера
	 */
	StepStorage* mSolverInfo;
	double mAtol; //solver absolute tolerance
	double mRtol; //solver relative tolerance

	/*
	 * Коммуникатор работников
	 * Может совпадать с MPI_COMM_WORLD, если нет питон-мастера
	 * либо это  MPI_COMM_WORLD без первого процесса
	 */
	MPI_Comm mWorkerComm;
	int mPythonMaster;

	/*
	 * Номер потока
	 */
	int mGlobalRank;
	int mWorkerRank;

	/*
	 * Количество потоков в целом
	 */
	int mWorkerCommSize;

	/*
	 * Глобальный Id задачи для базы
	 */
	//int mJobId;
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

	int dimension;

	int flags;

	int mStepCount;

	double startTime;
	double stopTime;

	double timeStep;
	int mAcceptedStepCount;
	int mRejectedStepCount;

	double currentTime;

	double saveInterval;
	double counterSaveTime;
	int mLastStepAccepted;

	double mDx, mDy, mDz;

	int mCellSize;
	int mHaloSize;

	int mRepeatCount;

	int totalGridNodeCount;
	int totalGridElementCount;

	MPI_Status status;

	//Solver* mPreviousState;

	void loadStateFromFile(char* dataFile);
	void setStopTime(double _stopTime);

	void readFileStat(std::ifstream& in);
	void readTimeSetting(std::ifstream& in);
	void readSaveInterval(std::ifstream& in);
	void readGridSteps(std::ifstream& in);
	void readCellAndHaloSize(std::ifstream& in);
	void readSolverIndex(std::ifstream& in);
	void readSolverTolerance(std::ifstream& in);
	void readBlockCount(std::ifstream& in);
	void readConnectionCount(std::ifstream& in);

	/*
	 * Чтение блока
	 */
	Block* readBlock(std::ifstream& in, int idx, int dimension);

	/*
	 * Чтение соединения
	 */
	Interconnect* readConnection(std::ifstream& in);

	void initSolvers();

	void prepareDeviceData(int deviceType, int deviceNumber, int stage);
	void processDeviceBlocksBorder(int deviceType, int deviceNumber, int stage);
	void processDeviceBlocksCenter(int deviceType, int deviceNumber, int stage);
	void prepareDeviceArgument(int deviceType, int deviceNumber, int stage);
	double getDeviceError(int deviceType, int deviceNumber);

	void prepareData(int stage);
	void computeOneStepBorder(int stage);
	void computeOneStepCenter(int stage);
	void prepareNextStageArgument(int stage);

	void computeStage(int stage);
	/*
	 * after every step (successful or not) we update timestep according to an error
	 */
	double collectError();

	void confirmStep();
	void rejectStep();

	int getMaximumNumberSavedStates();

	void createProcessigUnit();
	void createBlock(std::ifstream& in);
	void createInterconnect(std::ifstream& in);

	void initSolverInfo();

	void blockAfterCreate();

	Interconnect* getInterconnect(int sourceNode, int destinationNode, int borderLength, double* sourceData,
			double* destinationData);

	int getMaxStepStorageCount();
	int getElementCountOnProcessingUnit(int deviceType, int deviceNumber);
};

#endif /* SRC_DOMAIN_H_ */
