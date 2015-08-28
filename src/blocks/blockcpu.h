/*
 * BlockCpu.h
 *
 *  Created on: 20 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKCPU_H_
#define SRC_BLOCKCPU_H_

#include "block.h"

#include "../solvers/eulersolvercpu.h"
#include "../solvers/rk4solvercpu.h"
#include "../solvers/dp45solvercpu.h"

/*
 * Блок работы с данными на центральном процссоре.
 */

class BlockCpu: public Block {
private:
	void prepareBorder(int borderNumber, int stage, int zStart, int zStop, int yStart, int yStop, int xStart, int xStop);

	void computeStageCenter_1d(int stage, double time);
	void computeStageCenter_2d(int stage, double time);
	void computeStageCenter_3d(int stage, double time);

	void computeStageBorder_1d(int stage, double time);
	void computeStageBorder_2d(int stage, double time);
	void computeStageBorder_3d(int stage, double time);

	void createSolver(int solverIdx, double _aTol, double _rTol);

	double* getNewBlockBorder(Block* neighbor, int borderLength, int& memoryType);
	double* getNewExternalBorder(Block* neighbor, int borderLength, double* border, int& memoryType);

	void printSendBorderInfo();
	void printReceiveBorderInfo();
	void printParameters();
	void printComputeFunctionNumber();

public:
	BlockCpu(int _blockNumber, int _dimension, int _xCount, int _yCount, int _zCount,
			int _xOffset, int _yOffset, int _zOffset,
			int _nodeNumber, int _deviceNumber,
			int _haloSize, int _cellSize,
			unsigned short int* _initFuncNumber, unsigned short int* _compFuncNumber,
			int _solverIdx, double _aTol, double _rTol);

	~BlockCpu();

	bool isRealBlock() { return true; }

	int getBlockType() { return CPU; }

	void getCurrentState(double* result);

	void moveTempBorderVectorToBorderArray();
};

#endif /* SRC_BLOCKCPU_H_ */