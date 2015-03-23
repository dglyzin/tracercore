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

	/*
	 * Длина и ширина матрицы для вычислений.
	 */
	int length;
	int width;

	/*
	 * Координаты блока в области
	 */
	int lengthMove;
	int widthMove;

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

	/*
	 * Функция проверяет допустимость значений для данного блока
	 */
	bool checkValue(int side, int move);

	void freeMemory(int memory_alloc_type, double* memory);
	void freeMemory(int memory_alloc_type, int* memory);

public:
	Block(int _length, int _width, int _lengthMove, int _widthMove, int _nodeNumber, int _deviceNumber);
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

	virtual void swapMatrix() {
		double* tmp = matrix;
		matrix = newMatrix;
		newMatrix = tmp;
	}

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

	int getLength() { return length; }
	int getWidth() { return width; }

	/*
	 * Возвращает количество узлов области блока.
	 */
	int getCountGridNodes() { return length * width; }

	int getLengthMove() { return lengthMove; }
	int getWidthMove() { return widthMove; }

	int getDeviceNumber() { return deviceNumber; }

	int getNodeNumber() { return nodeNumber; }

	/*
	 * Вовращаение указателя на определенную границу.
	 * Выполняется проверка на существование этой границы.
	 */
	double* getTopBlockBorder() { return blockBorder != NULL ? blockBorder[TOP] : NULL; }
	double* getLeftBlockBorder() { return blockBorder != NULL ? blockBorder[LEFT] : NULL; }
	double* getBottomBlockBorder() { return blockBorder != NULL ? blockBorder[BOTTOM] : NULL; }
	double* getRightBlockBorder() { return blockBorder != NULL ? blockBorder[RIGHT] : NULL; }

	virtual double* addNewBlockBorder(Block* neighbor, int side, int move, int borderLength) { return NULL; }
	virtual double* addNewExternalBorder(Block* neighbor, int side, int move, int borderLength, double* border) { return NULL; }

	virtual void moveTempBorderVectorToBorderArray() { return; }

	virtual void loadData(double* data) { return; }
};

#endif /* SRC_BLOCK_H_ */
