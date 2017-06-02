/*
 * Domain.h
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_DOMAIN_H_
#define SRC_DOMAIN_H_

#include "blocks/nullblock.h"
#include "blocks/realblock.h"
#include "interconnect/transferinterconnect/transferinterconnectsend.h"
#include "interconnect/transferinterconnect/transferinterconnectrecv.h"
#include "interconnect/nontransferinterconnect.h"

#include "processingunit/cpu/cpu1d.h"
#include "processingunit/cpu/cpu2d.h"
#include "processingunit/cpu/cpu3d.h"

#include "processingunit/gpu/gpu1d.h"
#include "processingunit/gpu/gpu2d.h"
#include "processingunit/gpu/gpu3d.h"

#include "numericalmethod/euler.h"
#include "numericalmethod/rungekutta4.h"
#include "numericalmethod/dormandprince45.h"

#include "problem/ordinaryproblem.h"
#include "problem/delayproblem.h"

#include "utils.h"

#include <iostream>

/*
 * Основной управляющий класс приложения.
 */

class Domain {
public:
	Domain(int _world_rank, int _world_size, char* inputFile, char* binaryFileName, int _jobId);

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

	void saveStateForDraw(char* inputFile, int plotVals);
	void saveStateForLoad(char* inputFile);

	void saveStateForDrawDenseOutput(char* inputFile, double theta);

	void printStatisticsInfo(char* inputFile, char* outputFile, double calcTime, char* statisticsFile);

	bool isNan();

	void checkOptions(int flags, double _stopTime, char* saveFile);

	MPI_Comm getWorkerComm(){ return mWorkerComm;};

	int getUserStatus();
	int getEntirePlotValues();

private:
	ProcessingUnit* cpu;
	ProcessingUnit** gpu;

	int mProblemType;

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
	NumericalMethod* mNumericalMethod;
	double mAtol; //solver absolute tolerance
	double mRtol; //solver relative tolerance

	Problem* mProblem;

	/*
	 * Коммуникатор работников
	 * Всегда совпадает с MPI_COMM_WORLD
	 */
	MPI_Comm mWorkerComm;
	//int mPythonMaster;

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
	 * уникальный Id задачи для запросов состояния юзера  на сайте
	 */
	int mJobId;

	int mUserStatus;
	int mJobState;

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

	double mTimeStep;
	int mAcceptedStepCount;
	int mRejectedStepCount;

	double currentTime;

	double mSavePeriod;
	double mSaveTimer;
	int mLastStepAccepted;

	double mDx, mDy, mDz;

	int mCellSize;
	int mHaloSize;

	int mRepeatCount;

	int totalGridNodeCount;
	int totalGridElementCount;

	MPI_Status status;

	//system path to the folder containing hybriddomain and hybridsolver
	char mTracerFolder[250];
	char mProjectFolder[250];

	int mPlotCount;

	double* mPlotPeriods;
	double* mPlotTimers;

	bool isRealTimePNG;

	//Solver* mPreviousState;

	void loadStateFromFile(char* dataFile);
	void setStopTime(double _stopTime);

	void readFileStat(std::ifstream& in);
	void readTimeSetting(std::ifstream& in);
	void readSavePeriod(std::ifstream& in);
	void readGridSteps(std::ifstream& in);
	void readCellAndHaloSize(std::ifstream& in);
	void readSolverIndex(std::ifstream& in);
	void readSolverTolerance(std::ifstream& in);
	void readBlockCount(std::ifstream& in);
	void readPlotCount(std::ifstream& in);
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
	void prepareStageArgument(int stage);

	void prepareBlockStageData(int stage);


	void computeStage(int stage);
	/*
	 * after every step (successful or not) we update timestep according to an error
	 */
	double collectError();
	bool isErrorPermissible(double error);
	double computeNewStep(double error);

	void confirmStep();
	void rejectStep();

	void createProcessigUnit();
	void createBlock(std::ifstream& in);
	void createInterconnect(std::ifstream& in);

	void readPlots(std::ifstream& in);

	void createNumericalMethod();
	void createProblem();

	void blockAfterCreate();

	Interconnect* getInterconnect(int sourceNode, int destinationNode, int borderLength, double* sourceData,
			double* destinationData);

	int getMaxStepStorageCount();
	int getRequiredMemoryOnProcessingUnit(int deviceType, int deviceNumber);
	void saveGeneralInfo(char* path);
	void saveStateForDrawByBlocks(char* path);
	void saveStateForLoadByBlocks(char* path);
	void saveStateForDrawDenseOutputByBlocks(char* path, double theta);

	double getThetaForDenseOutput(double requiredTime);
	int isReadyToFullSave();
	int isReadyToPlot();

	void stopByUser(char* inputFile);
	void stopByTime(char* inputFile);

	void fixInitialBorderValues(int sourceBlock, int destinationBlock, int* offsetSource, int* offsetDestination, int* length, int sourceSide, int destinationSide);
};

#endif /* SRC_DOMAIN_H_ */
