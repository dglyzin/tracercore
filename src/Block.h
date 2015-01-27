/*
 * Block.h
 *
 *  Created on: 17 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCK_H_
#define SRC_BLOCK_H_

#include <stdio.h>

#include "Enums.h"

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
	 * Координаты блока в области
	 */
	int lenghtMove;
	int widthMove;

	/*
	 * Номер потока исполнения, на котором работает этот блок
	 */
	int nodeNumber;

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
	Block(int _length, int _width, int _lengthMove, int _widthMove, int _nodeNumber);
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
	virtual void courted(double dX, double dY) { return; }

	/*
	 * Возвращает тип блока.
	 */
	virtual int getBlockType() { return CPU; }

	/*
	 * Печатает информацию о блоке на консоль.
	 */
	virtual void print(int locationNode) { return; }

	/*
	 * Печатает только матрицу
	 */
	virtual void printMatrix() { return; }

	virtual double** getResault() { return matrix; }

	int getLength() { return length; }
	int getWidth() { return width; }

	int getLenghtMove() { return lenghtMove; }
	int getWidthMove() { return widthMove; }

	int getNodeNumber() { return nodeNumber; }

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

	virtual void setTopExternalBorder(double* _topExternalBorder) { return; }
	virtual void setLeftExternalBorder(double* _leftExternalBorder) { return; }
	virtual void setBottomExternalBorder(double* _borderExternalBorder) { return; }
	virtual void setRightExternalBorder(double* _rightExternalBorder) { return; }

	/*virtual void createTopBorderType() { topBorderType = NULL; }
	virtual void createLeftBorderType() { leftBorderType = NULL; }
	virtual void createBottomBorderType() { bottomBorderType = NULL; }
	virtual void createRightBorderType() { rightBorderType = NULL; }

	void createBorderType() {
		createTopBorderType();
		createLeftBorderType();
		createBottomBorderType();
		createRightBorderType();
	}*/

	virtual void createTopBlockBorder(int neighborType) { topBlockBorder = NULL; }
	virtual void createLeftBlockBorder(int neighborType) { leftBlockBorder = NULL; }
	virtual void createBottomBlockBorder(int neighborType) { bottomBlockBorder = NULL; }
	virtual void createRightBlockBorder(int neighborType) { rightBlockBorder = NULL; }

	void createBlockBorder( int topNeighborType, int leftNeighborType, int bottonNeighborType, int rightNeighborType ) {
		createTopBlockBorder(topNeighborType);
		createLeftBlockBorder(leftNeighborType);
		createBottomBlockBorder(bottonNeighborType);
		createRightBlockBorder(rightNeighborType);
	}
/*
	virtual void createTopExteranalBroder() { topExternalBorder = NULL; }
	virtual void createLeftExternalBorder() { leftExternalBorder = NULL; }
	virtual void createBottomExternalBorder() { bottomExternalBorder = NULL; }
	virtual void createRightExternalBorder() { rightExternalBorder = NULL; }

	void createExternalBorder() {
		createTopExteranalBroder();
		createLeftExternalBorder();
		createBottomExternalBorder();
		createRightExternalBorder();
	}*/
};

#endif /* SRC_BLOCK_H_ */
