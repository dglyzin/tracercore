/*
 * BlockGpu.h
 *
 *  Created on: 29 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKGPU_H_
#define SRC_BLOCKGPU_H_

#include "Block.h"

#define BLOCK_LENGHT_SIZE 32
#define BLOCK_WIDTH_SIZE 16

#define BLOCK_SIZE 512

/*
 * Класс обработки данных на видеокарте
 */

class BlockGpu: public Block {
private:
	int deviceNumber;

	int** borderTypeOnDevice;
	double** blockBorderOnDevice;
	double** externalBorderOnDevice;

public:
	BlockGpu(int _length, int _width, int _lengthMove, int _widthMove, int _world_rank, int _deviceNumber);
	virtual ~BlockGpu();

	bool isRealBlock() { return true; }

	void prepareData();

	void courted(double dX2, double dY2, double dT);

	int getBlockType();

	void setPartBorder(int type, int side, int move, int borderLength);

	double* getResault();

	void print();
};

#endif /* SRC_BLOCKGPU_H_ */
