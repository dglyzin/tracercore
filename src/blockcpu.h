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
			unsigned short int* _initFuncNumber, unsigned short int* _compFuncNumber,
			int _solverIndex);

	~BlockCpu();

	bool isRealBlock() { return true; }

	void prepareStageData(int stage);

	void computeStageBorder(int stage, double time, double step) {}// std::cout << std::endl << "one step border" << std::endl; }
	void computeStageCenter(int stage, double time, double step) {}// std::cout << std::endl << "one step center" << std::endl; }

	int getBlockType() { return CPU; }

	double* getCurrentState(double* result);

	void print();

	double* addNewBlockBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength);
	double* addNewExternalBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength, double* border);

	void moveTempBorderVectorToBorderArray();

	void loadData(double* data);

	void prepareLeftBorder(double* source, int borderNumber, int mOffset, int nOffset, int mLength, int nLength);
	void prepareRightBorder(double* source, int borderNumber, int mOffset, int nOffset, int mLength, int nLength);
	void prepareFrontBorder(double* source, int borderNumber, int mOffset, int nOffset, int mLength, int nLength);
	void prepareBackBorder(double* source, int borderNumber, int mOffset, int nOffset, int mLength, int nLength);
	void prepareTopBorder(double* source, int borderNumber, int mOffset, int nOffset, int mLength, int nLength);
	void prepareBottomBorder(double* source, int borderNumber, int mOffset, int nOffset, int mLength, int nLength);

	void prepareBorder(double* source, int borderNumber, int zStart, int zStop, int yStart, int yStop, int xStart, int xStop);

	Solver* createSolver(int solverIdx, int count);
};

#endif /* SRC_BLOCKCPU_H_ */
