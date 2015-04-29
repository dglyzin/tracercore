/*
 * BlockNull.h
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKNULL_H_
#define SRC_BLOCKNULL_H_

#include "block.h"

/*
 * Блок - загушка.
 * Отвечает false на вопрос о своей реальности.
 * Остальные функции своей предка не переопределяет.
 */

class BlockNull: public Block {
private:
	void prepareBorder(double* source, int borderNumber, int zStart, int zStop, int yStart, int yStop, int xStart, int xStop) { return; }

	void computeStageCenter_1d(int stage, double time, double step) { return; }
	void computeStageCenter_2d(int stage, double time, double step) { return; }
	void computeStageCenter_3d(int stage, double time, double step) { return; }

	void computeStageBorder_1d(int stage, double time, double step) { return; }
	void computeStageBorder_2d(int stage, double time, double step) { return; }
	void computeStageBorder_3d(int stage, double time, double step) { return; }

public:
	BlockNull(int _dimension, int _xCount, int _yCount, int _zCount,
				int _xOffset, int _yOffset, int _zOffset,
				int _nodeNumber, int _deviceNumber,
				int _haloSize, int _cellSize);
	virtual ~BlockNull();

	bool isRealBlock() { return false; }

	void prepareStageData(int stage) { return; }

	double getSolverStepError(double timeStep, double aTol, double rTol) { return 0.0; }

	int getBlockType() { return NULL_BLOCK; }

	void getCurrentState(double* result) { return; }

	void print() { return; }

	double* addNewBlockBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength) { return NULL; }
	double* addNewExternalBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength, double* border) { return NULL; }

	void moveTempBorderVectorToBorderArray() { return; }

	void loadData(double* data) { return; }

	void createSolver(int solverIdx) { return; }
};

#endif /* SRC_BLOCKNULL_H_ */
