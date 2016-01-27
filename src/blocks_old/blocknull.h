/*
 * BlockNull.h
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKNULL_H_
#define SRC_BLOCKNULL_H_

#include "../blocks_old/block.h"

/*
 * Блок - загушка.
 * Отвечает false на вопрос о своей реальности.
 * Остальные функции своей предка не переопределяет.
 */

class BlockNull: public Block_old {
private:
	void prepareBorder(int borderNumber, int stage, int zStart, int zStop, int yStart, int yStop, int xStart, int xStop) { return; }

	void createSolver(int solverIdx, double _aTol, double _rTol) { return; }

	double* getNewBlockBorder(Block_old* neighbor, int borderLength, int& memoryType) { return NULL; }
	double* getNewExternalBorder(Block_old* neighbor, int borderLength, double* border, int& memoryType) { return NULL; }

	void printSendBorderInfo() { return; }
	void printReceiveBorderInfo() { return; }
	void printParameters() { return; }
	void printComputeFunctionNumber() { return; }

public:
	BlockNull(int _blockNumber, int _dimension, int _xCount, int _yCount, int _zCount,
			int _xOffset, int _yOffset, int _zOffset,
			int _nodeNumber, int _deviceNumber,
			int _haloSize, int _cellSize);

	virtual ~BlockNull();

	bool isRealBlock() { return false; }

	int getBlockType() { return NULL_BLOCK; }

	void getCurrentState(double* result) { return; }

	void moveTempBorderVectorToBorderArray() { return; }

	void computeStageBorder(int stage, double time) { return; }
	void computeStageCenter(int stage, double time) { return; }
};

#endif /* SRC_BLOCKNULL_H_ */
