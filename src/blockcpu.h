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
private:
	void prepareBorder(double* source, int borderNumber, int zStart, int zStop, int yStart, int yStop, int xStart, int xStop);

	void computeStageCenter_1d(int stage, double time, double step);
	void computeStageCenter_2d(int stage, double time, double step);
	void computeStageCenter_3d(int stage, double time, double step);

	void computeStageBorder_1d(int stage, double time, double step);
	void computeStageBorder_2d(int stage, double time, double step);
	void computeStageBorder_3d(int stage, double time, double step);

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

	double getSolverStepError(double timeStep, double aTol, double rTol);

	int getBlockType() { return CPU; }

	double* getCurrentState(double* result);

	void print();

	double* addNewBlockBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength);
	double* addNewExternalBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength, double* border);

	void moveTempBorderVectorToBorderArray();

	void loadData(double* data);

	Solver* createSolver(int solverIdx, int count);
};

#endif /* SRC_BLOCKCPU_H_ */
