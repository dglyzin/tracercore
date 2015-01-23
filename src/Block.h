/*
 * Block.h
 *
 *  Created on: 17 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCK_H_
#define SRC_BLOCK_H_

#include <stdio.h>

/*
 * Типы границ блока
 */
enum BORDER_TYPE {BY_ANOTHER_BLOCK, BY_FUNCTION};

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
	double** matrix;

	/*
	 * Длина и ширина матрицы для вычислений.
	 */
	int length;
	int width;

	/*
	 * Тип границы блока.
	 * BY_ANOTHER_BLOCK - граница с другим блоком, работает через Interconnect.
	 * BY_FUNCTION - границы с другим блоком нет, значения даются функцией.
	 */
	int* topBorderType;
	int* leftBorderType;
	int* bottomBorderType;
	int* rightBorderType;

	/*
	 * Граничные условия для других блоков,
	 * сюда блок самостоятельно укладывает свежие данныепосле каждой итерации.
	 * Interconnect их забирает (должен знать откуда забирать).
	 */
	double* topBlockBorder;
	double* leftBlockBorder;
	double* bottomBlockBorder;
	double* rightBlockBorder;

	/*
	 * С помощью Interconnect'а здесь будут находится свежие данные от других блоков,
	 * кроме того, сюда же записывают данные граничные функции.
	 * Первыми пишут Interconnect'ы, затем функции.
	 */
	double* topExternalBorder;
	double* leftExternalBorder;
	double* bottomExternalBorder;
	double* rightExternalBorder;

	/*
	 * TODO
	 * В этом классе (и в его потомках) необходимо учесть наличие функции расчета,
	 * которая принимает исходную матрицу и что-то еще.
	 * Пока не очень ясно что именно.
	 */

public:
	Block();
	Block(int _length, int _width);
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
	 * TODO
	 * Реализовать работу по заданной из вне функции.
	 */
	virtual void courted() { return; }

	/*
	 * Печатает информацию о блоке на консоль.
	 */
	virtual void print(int locationNode) { return; }

	/*
	 * Печатает только матрицу
	 */
	virtual void printMatrix() { return; }

	virtual double** getResault() { return matrix; }

	int* getTopBorderType() { return topBorderType; }
	int* getLeftBorderType() { return leftBorderType; }
	int* getBottomBorderType() { return bottomBorderType; }
	int* getRightBorderType() { return rightBorderType; }

	double* getTopBlockBorder() { return topBlockBorder; }
	double* getLeftBlockBorder() { return leftBlockBorder; }
	double* getBottomBlockBorder() { return bottomBlockBorder; }
	double* getRightBlockBorder() { return rightBlockBorder; }

	double* getTopExternalBorder() { return topExternalBorder; }
	double* getLeftExternalBorder() { return leftExternalBorder; }
	double* getBottomExternalBorder() { return bottomExternalBorder; }
	double* getRightExternalBorder() { return rightExternalBorder; }
};

#endif /* SRC_BLOCK_H_ */
