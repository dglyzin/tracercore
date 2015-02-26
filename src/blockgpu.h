/*
 * BlockGpu.h
 *
 *  Created on: 29 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKGPU_H_
#define SRC_BLOCKGPU_H_

#include "block.h"

#define BLOCK_LENGHT_SIZE 32
#define BLOCK_WIDTH_SIZE 16

#define BLOCK_SIZE 512

/*
 * Класс обработки данных на видеокарте
 */

class BlockGpu: public Block {
private:
	int** sendBorderTypeOnDevice;
	int** receiveBorderTypeOnDevice;

	double** blockBorderOnDevice;

	double** externalBorderOnDevice;

public:
	BlockGpu(int _length, int _width, int _lengthMove, int _widthMove, int _nodeNumber, int _deviceNumber);
	virtual ~BlockGpu();

	bool isRealBlock() { return true; }

	void prepareData();

	void computeOneStep(double dX2, double dY2, double dT);

	int getBlockType() { return GPU; }

	double* getResult();

	void print();

	double* addNewBlockBorder(Block* neighbor, int side, int move, int borderLength);
	double* addNewExternalBorder(Block* neighbor, int side, int move, int borderLength, double* border);

	void moveTempBorderVectorToBorderArray();

	void loadData(double* data);
};

#endif /* SRC_BLOCKGPU_H_ */
