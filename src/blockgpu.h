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
	double** blockBorderOnDevice;

	double** externalBorderOnDevice;

public:
	BlockGpu(int _dimension, int _xCount, int _yCount, int _zCount,
			int _xOffset, int _yOffset, int _zOffset,
			int _nodeNumber, int _deviceNumber,
			int _haloSize, int _cellSize,
			unsigned short int* _initFuncNumber, unsigned short int* _compFuncNumber,
			int _mSolverIndex);
	virtual ~BlockGpu();

	bool isRealBlock() { return true; }

	void prepareStageData(int stage) { std::cout << std::endl << "prepare data" << std::endl; }

	void computeStageBorder(int stage, double time, double step) { std::cout << std::endl << "one step border" << std::endl; }
	void computeStageCenter(int stage, double time, double step) { std::cout << std::endl << "one step center" << std::endl; }

	int getBlockType() { return GPU; }

	void getCurrentState(double* result);

	void print();

	double* addNewBlockBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength) { std::cout << std::endl << "add new block border" << std::endl; return NULL; }
	double* addNewExternalBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength, double* border) { std::cout << std::endl << "add new external border" << std::endl; return NULL; }

	void moveTempBorderVectorToBorderArray() { std::cout << std::endl << "move array to vector" << std::endl; }

	void loadData(double* data);
};

#endif /* SRC_BLOCKGPU_H_ */
