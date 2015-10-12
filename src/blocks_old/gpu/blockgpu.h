/*
 * BlockGpu.h
 *
 *  Created on: 29 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKGPU_H_
#define SRC_BLOCKGPU_H_

#include "../block.h"

#include "../../solvers/euler/eulersolvergpu.h"
#include "../../solvers/rk4/rk4solvergpu.h"
#include "../../solvers/dp45/dp45solvergpu.h"

/*
 * Класс обработки данных на видеокарте
 */

class BlockGpu: public Block_old {
private:
	void prepareBorder(int borderNumber, int stage, int zStart, int zStop, int yStart, int yStop, int xStart, int xStop);

	void createSolver(int solverIdx, double _aTol, double _rTol);

	double* getNewBlockBorder(Block_old* neighbor, int borderLength, int& memoryType);
	double* getNewExternalBorder(Block_old* neighbor, int borderLength, double* border, int& memoryType);

	void printSendBorderInfo();
	void printReceiveBorderInfo();
	void printParameters();
	void printComputeFunctionNumber();

public:
	BlockGpu(int _blockNumber, int _dimension, int _xCount, int _yCount, int _zCount,
			int _xOffset, int _yOffset, int _zOffset,
			int _nodeNumber, int _deviceNumber,
			int _haloSize, int _cellSize,
			unsigned short int* _initFuncNumber, unsigned short int* _compFuncNumber,
			int _solverIdx, double _aTol, double _rTol);

	virtual ~BlockGpu();

	bool isRealBlock() { return true; }

	int getBlockType() { return GPU; }

	void getCurrentState(double* result);

	void moveTempBorderVectorToBorderArray();
};

#endif /* SRC_BLOCKGPU_H_ */
