/*
 * Block.h
 *
 *  Created on: 17 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCK_H_
#define SRC_BLOCK_H_

#include <stdio.h>

enum BORDER_TYPE {BY_ANOTHER_BLOCK, BY_FUNCTION};
//#define BY_ANOTHER_BLOCK 0
//#define BY_FUNCTION 1

class Block {

protected:
	// матрица для вычислений
	double** matrix;

	// дина и ширина матрицы для вычислений
	int length;
	int width;

	// тип границы блока
	// 0 - граница с другим блоком, работает через Interconnect
	// 1 - .. - границы с другим блоком нет, значения даются функцией
	int* topBorderType;
	int* leftBorderType;
	int* bottomBorderType;
	int* rightBorderType;

	// граничные условия для других блоков,
	// сюда блок самостоятельно укладывает свежие данные
	// после каждой итерации.
	// Interconnect их забирает (должен знать откуда забирать)
	double* topBlockBorder;
	double* leftBlockBorder;
	double* bottomBlockBorder;
	double* rightBlockBorder;

	// с помощью Interconnect'а здесь будут находится свежие данные от других блоков,
	// кроме того, сюда же записывают данные граничные функции
	double* topExternalBorder;
	double* leftExternalBorder;
	double* bottomExternalBorder;
	double* rightExternalBorder;

	//ФУНКЦИИ!!!!!!!!!!

public:
	Block();
	Block(int _length, int _width);
	virtual ~Block();

	virtual bool isRealBlock() { return false; }
	virtual void prepareData() { return; }
	virtual void courted() { return; }

	virtual void print(int locationNode) { return; }
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
