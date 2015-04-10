/*
 * BlockCpu.h
 *
 *  Created on: 20 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKCPU_H_
#define SRC_BLOCKCPU_H_

#include "block.h"

/*
 * Блок работы с данными на центральном процссоре.
 */

class BlockCpu: public Block {
public:
	BlockCpu(int _dimension, int _xCount, int _yCount, int _zCount,
			int _xOffset, int _yOffset, int _zOffset,
			int _nodeNumber, int _deviceNumber,
			int _haloSize, int _cellSize,
			unsigned short int* _initFuncNumber, unsigned short int* _compFuncNumber);

	~BlockCpu();

	bool isRealBlock() { return true; }

	void prepareStageData(int stage);

	void computeStageBorder(double* result, double* source, double time, double* param) { std::cout << std::endl << "one step border" << std::endl; }
	void computeStageCenter(double* result, double* source, double time, double* param) { std::cout << std::endl << "one step center" << std::endl; }

	int getBlockType() { return CPU; }

	double* getCurrentState();

	void print();

	double* addNewBlockBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength);
	double* addNewExternalBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength, double* border);

	void moveTempBorderVectorToBorderArray();

	void loadData(double* data);
};

#endif /* SRC_BLOCKCPU_H_ */
