/*
 * BlockGpu.h
 *
 *  Created on: 29 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKGPU_H_
#define SRC_BLOCKGPU_H_

#include "block.h"

#include "../solvers/eulersolvergpu.h"
#include "../solvers/rk4solvergpu.h"
#include "../solvers/dp45solvergpu.h"

/*
 * Класс обработки данных на видеокарте
 */

class BlockGpu: public Block {
private:
	/*double** blockBorderOnDevice;

	double** externalBorderOnDevice;*/

	void prepareBorder(int borderNumber, int stage, int zStart, int zStop, int yStart, int yStop, int xStart, int xStop);

	void computeStageCenter_1d(int stage, double time) { std::cout << std::endl << "GPU compute center 1d" << std::endl; }
	void computeStageCenter_2d(int stage, double time) { std::cout << std::endl << "GPU compute center 2d" << std::endl; }
	void computeStageCenter_3d(int stage, double time) { std::cout << std::endl << "GPU compute center 3d" << std::endl; }

	void computeStageBorder_1d(int stage, double time) { std::cout << std::endl << "GPU compute border 1d" << std::endl; }
	void computeStageBorder_2d(int stage, double time) { std::cout << std::endl << "GPU compute border 2d" << std::endl; }
	void computeStageBorder_3d(int stage, double time) { std::cout << std::endl << "GPU compute border 3d" << std::endl; }

	void createSolver(int solverIdx, double _aTol, double _rTol);

	double* getNewBlockBorder(Block* neighbor, int borderLength, int& memoryType);
	double* getNewExternalBorder(Block* neighbor, int borderLength, double* border, int& memoryType);

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

	//void prepareStageData(int stage) { std::cout << std::endl << "GPU prepare data" << std::endl; }

	int getBlockType() { return GPU; }

	void getCurrentState(double* result);

	//void print();

	//double* addNewBlockBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength); //{ std::cout << std::endl << "GPU add new block border" << std::endl; return NULL; }
	//double* addNewExternalBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength, double* border) { std::cout << std::endl << "GPU add new external border" << std::endl; return NULL; }

	void moveTempBorderVectorToBorderArray();// { std::cout << std::endl << "GPU move array to vector" << std::endl; }
};

#endif /* SRC_BLOCKGPU_H_ */
