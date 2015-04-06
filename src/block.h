/*
 * Block.h
 *
 *  Created on: 17 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCK_H_
#define SRC_BLOCK_H_

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <math.h>
#include <string.h>
#include <iostream>

#include <vector>

#include <omp.h>

#include "enums.h"

/*
 * Класс, отвечающий за обработку данных.
 * Является родителем для классов, реально использующихся для вычислений.
 * Наследники: BlockCpu, BlockGpu, BlockNull
 */

class Block {

protected:
	/*
	 * Матрица для вычислений.
	 * Хранит текущее значения области.
	 * Из нее получаются границы блока - для пересылки
	 */
	double* matrix;
	double* newMatrix;

	int dimension;

	int xCount;
	int yCount;
	int zCount;

	int xOffset;
	int yOffset;
	int zOffset;

	unsigned short int* functionNumber;

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

	/*
	 * Тип границы блока.
	 * BY_FUNCTION - границы с другим блоком нет, значения даются функцией.
	 */
	int** sendBorderType;
	int** receiveBorderType;

	/*
	 * Граничные условия для других блоков,
	 * сюда блок самостоятельно укладывает свежие данныепосле каждой итерации.
	 * Interconnect их забирает (должен знать откуда забирать).
	 */
	double** blockBorder;
	int* blockBorderMove;
	int* blockBorderMemoryAllocType;
	std::vector<double*> tempBlockBorder;
	std::vector<int> tempBlockBorderMove;
	std::vector<int> tempBlockBorderMemoryAllocType;

	/*
	 * С помощью Interconnect'а здесь будут находится свежие данные от других блоков,
	 * кроме того, сюда же записывают данные граничные функции.
	 * Первыми пишут Interconnect'ы, затем функции.
	 */
	double** externalBorder;
	int* externalBorderMove;
	int* externalBorderMemoryAllocType;
	std::vector<double*> tempExternalBorder;
	std::vector<int> tempExternalBorderMove;
	std::vector<int> tempExternalBorderMemoryAllocType;


	//double* result;

	/*
	 * Количество частей гранцы для пересылки и для получения
	 */
	int countSendSegmentBorder;
	int countReceiveSegmentBorder;

	void freeMemory(int memory_alloc_type, double* memory);
	void freeMemory(int memory_alloc_type, int* memory);

public:
	Block(int _dimension, int _xCount, int _yCount, int _zCount,
			int _xOffset, int _yOffset, int _zOffset,
			int _nodeNumber, int _deviceNumber,
			int _haloSize, int _cellSize);

	virtual ~Block();

	/*
	 * Проверяет, является ли блок реальным для данного потока исполнения.
	 * true - да, является
	 * false - нет не является
	 */
	virtual bool isRealBlock() { return false; }

	/*
	 * Выполняет подготовку данных.
	 * Заполняет массивы границ для пересылки.
	 */
	virtual void prepareData() { return; }

	/*
	 * Выполняет вычисления.
	 */
	virtual void computeOneStep(double dX2, double dY2, double dT) { return; }

	virtual void computeOneStepBorder(double dX2, double dY2, double dT) { return; }
	virtual void computeOneStepCenter(double dX2, double dY2, double dT) { return; }

	void swapMatrix();

	/*
	 * Возвращает тип блока.
	 */
	virtual int getBlockType() { return NULL_BLOCK; }

	/*
	 * Печатает информацию о блоке на консоль.
	 */
	virtual void print() { return; }

	/*
	 * Возвращает результурющую матрицу данного блока.
	 */
	virtual double* getCurrentState() { return NULL; }

	int getXCount() { return xCount; }
	int getYCount() { return yCount; }
	int getZCount() { return zCount; }

	int getXOffset() { return xOffset; }
	int getYOffset() { return yOffset; }
	int getZOffset() { return zOffset; }

	int getGridNodeCount();

	int getDeviceNumber() { return deviceNumber; }
	int getNodeNumber() { return nodeNumber; }

	void setFunctionNumber(unsigned short int* functionNumberData ) { return; }

	virtual double* addNewBlockBorder(Block* neighbor, int side, int move, int borderLength) { return NULL; }
	virtual double* addNewExternalBorder(Block* neighbor, int side, int move, int borderLength, double* border) { return NULL; }

	virtual void moveTempBorderVectorToBorderArray() { return; }

	virtual void loadData(double* data) { return; }
};

#endif /* SRC_BLOCK_H_ */
