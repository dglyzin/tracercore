/*
 * Block.h
 *
 *  Created on: 17 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCK_H_
#define SRC_BLOCK_H_

#include <vector>

#include <omp.h>

#include "../enums.h"
#include "../solvers/solver.h"
#include "../userfuncs.h"

/*
 * Класс, отвечающий за обработку данных.
 * Является родителем для классов, реально использующихся для вычислений.
 * Наследники: BlockCpu, BlockGpu, BlockNull
 */

class Block {

protected:
	Solver* mSolver;

	int blockNumber;

	int dimension;

	int xCount;
	int yCount;
	int zCount;

	int xOffset;
	int yOffset;
	int zOffset;

	unsigned short int* mCompFuncNumber;

	int cellSize;
	int haloSize;

	/*
	 * Тип устройства.
	 * Для видеокарт - номер видеокарты.
	 * Для ЦПУ - предполагается номер сокета.
	 */
	int deviceNumber;

	/*
	 * Номер потока исполнения, на котором работает этот блок
	 * Номер потока, который ДОЛЖЕН его создать для работы.
	 * Номер потока, на котором это блок РЕАЛЬНО сущесвтует.
	 */
	int nodeNumber;

	int* sendBorderInfo;
	std::vector<int> tempSendBorderInfo;

	int* receiveBorderInfo;
	std::vector<int> tempReceiveBorderInfo;

	/*
	 * Граничные условия для других блоков,
	 * сюда блок самостоятельно укладывает свежие данныепосле каждой итерации.
	 * Interconnect их забирает (должен знать откуда забирать).
	 */
	double** blockBorder;
	int* blockBorderMemoryAllocType;
	std::vector<double*> tempBlockBorder;
	std::vector<int> tempBlockBorderMemoryAllocType;

	/*
	 * С помощью Interconnect'а здесь будут находится свежие данные от других блоков,
	 * кроме того, сюда же записывают данные граничные функции.
	 * Первыми пишут Interconnect'ы, затем функции.
	 */
	double** externalBorder;
	int* externalBorderMemoryAllocType;
	std::vector<double*> tempExternalBorder;
	std::vector<int> tempExternalBorderMemoryAllocType;

	/*
	 * Количество частей гранцы для пересылки и для получения
	 */
	int countSendSegmentBorder;
	int countReceiveSegmentBorder;

	void freeMemory(int memory_alloc_type, double* memory);
	void freeMemory(int memory_alloc_type, int* memory);

	virtual void prepareBorder(int borderNumber, int stage, int zStart, int zStop, int yStart, int yStop, int xStart, int xStop) = 0;

	virtual void computeStageCenter_1d(int stage, double time) = 0;
	virtual void computeStageCenter_2d(int stage, double time) = 0;
	virtual void computeStageCenter_3d(int stage, double time) = 0;

	virtual void computeStageBorder_1d(int stage, double time) = 0;
	virtual void computeStageBorder_2d(int stage, double time) = 0;
	virtual void computeStageBorder_3d(int stage, double time) = 0;

	virtual void createSolver(int solverIdx, double _aTol, double _rTol) = 0;

	virtual double* getNewBlockBorder(Block* neighbor, int borderLength, int& memoryType) = 0;
	virtual double* getNewExternalBorder(Block* neighbor, int borderLength, double* border, int& memoryType) = 0;

	virtual void printSendBorderInfo() = 0;
	virtual void printReceiveBorderInfo() = 0;
	virtual void printParameters() = 0;
	virtual void printComputeFunctionNumber() = 0;

	void printSendBorderInfoArray(int* sendBorderInfoArray);
	void printReceiveBorderInfoArray(int* recieveBorderInfoArray);

	func_ptr_t* mUserFuncs;
	initfunc_fill_ptr_t* mUserInitFuncs;
	double* mParams;
	int mParamsCount;

public:
	Block(int _blockNumber, int _dimension, int _xCount, int _yCount, int _zCount,
			int _xOffset, int _yOffset, int _zOffset,
			int _nodeNumber, int _deviceNumber,
			int _haloSize, int _cellSize);

	virtual ~Block();

	/*
	 * Проверяет, является ли блок реальным для данного потока исполнения.
	 * true - да, является
	 * false - нет не является
	 */
	virtual bool isRealBlock() = 0;

	virtual void initSolver() { return; }

	/*
	 * Выполняет подготовку данных.
	 * Заполняет массивы границ для пересылки.
	 */
	void prepareStageData(int stage);

	void computeStageBorder(int stage, double time);
	void computeStageCenter(int stage, double time);

	void prepareArgument(int stage, double timestep );

	double getSolverStepError(double timeStep) { return mSolver != NULL ? mSolver->getStepError(timeStep) : 0.0; }

	void confirmStep(double timestep);
	void rejectStep(double timestep);

	/*
	 * Возвращает тип блока.
	 */
	virtual int getBlockType() = 0;

	/*
	 * Печатает информацию о блоке на консоль.
	 */
	void print();
	void printGeneralInformation();

	/*
	 * Возвращает результурющую матрицу данного блока.
	 */
	virtual void getCurrentState(double* result) = 0;

	int getXCount() { return xCount; }
	int getYCount() { return yCount; }
	int getZCount() { return zCount; }

	int getXOffset() { return xOffset; }
	int getYOffset() { return yOffset; }
	int getZOffset() { return zOffset; }

	int getGridNodeCount();
	int getGridElementCount();

	int getDeviceNumber() { return deviceNumber; }
	int getNodeNumber() { return nodeNumber; }

	int getDimension() { return dimension; }
	
	void setFunctionNumber(unsigned short int* functionNumberData ) { return; }

	double* addNewBlockBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength);
	double* addNewExternalBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength, double* border);

	virtual void moveTempBorderVectorToBorderArray() = 0;

	void loadData(double* data);
};

#endif /* SRC_BLOCK_H_ */
