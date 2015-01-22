/*
 * Block.h
 *
 *  Created on: 17 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCK_H_
#define SRC_BLOCK_H_

#include <stdio.h>

class Block {

protected:
	// номер узла, на котором расположен данный блок
	//int numberLocationNode;

	// тип блока. ЦПУ или Видеокарта (+ее номер)
	//int blockType;

	// матрица для вычислений
	double** matrix;

	// дина и ширина матрицы для вычислений
	int length;
	int width;

	// тип границы блока
	// 0 - граница с другим блоком, работает через Interconnect
	// 1 - .. - границы с другим блоком нет, значения даются функцией
	int* topBoundaryType;
	int* leftBoundaryType;
	int* bottomBoundaryType;
	int* rightBoundaryType;

	// граничные условия для других блоков,
	// сюда блок самостоятельно укладывает свежие данные
	// после каждой итерации.
	// Interconnect их забирает (должен знать откуда забирать)
	double* topBlockBoundary;
	double* leftBlockBoundary;
	double* bottomBlockBoundary;
	double* rightBlockBoundary;

	// с помощью Interconnect'а здесь будут находится свежие данные от других блоков,
	// кроме того, сюда же записывают данные граничные функции
	double* topExternalBoundary;
	double* leftExternalBoundary;
	double* bottomExternalBoundary;
	double* rightExternalBoundary;

	//ФУНКЦИИ!!!!!!!!!!

public:
	Block();
	Block(int _length, int _width,
			int* _topBoundaryType, int* _leftBoundaryType, int* _bottomBoundaryType, int* _rightBoundaryType,
			double* _topBlockBoundary, double* _leftBlockBoundary, double* _bottomBlockBoundary, double* _rightBlockBoundary,
			double* _topExternalBoundary, double* _leftExternalBoundary, double* _bottomExternalBoundary, double* _rightExternalBoundary);
	virtual ~Block();

	virtual bool isRealBlock() { return false; }
	virtual void prepareData() { return; }
	virtual void courted() { return; }

	virtual void print(int locationNode) { return; }
	virtual void printMatrix() { return; }

	virtual double** getResault() { return matrix; }
};

#endif /* SRC_BLOCK_H_ */
