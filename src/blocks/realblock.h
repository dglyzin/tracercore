/*
 * realblock.h
 *
 *  Created on: 15 окт. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKS_REALBLOCK_H_
#define SRC_BLOCKS_REALBLOCK_H_

#include <vector>

#include "../problem/ordinary.h"
#include "../problem/delay.h"

#include "../processingunit/processingunit.h"

#include "block.h"

class RealBlock: public Block {
public:
	RealBlock(int _nodeNumber, int _dimension,
			int _xCount, int _yCount, int _zCount,
			int _xOffset, int _yOffset, int _zOffset,
			int _cellSize, int _haloSize,
			int _blockNumber, ProcessingUnit* _pu,
			unsigned short int* _initFuncNumber, unsigned short int* _compFuncNumber,
			int problemType, int solverType, double aTol, double rTol);

	virtual ~RealBlock();

	void computeStageBorder(int stage, double time);
	void computeStageCenter(int stage, double time);

	void prepareArgument(int stage, double timestep );

	void prepareStageData(int stage);


	bool isRealBlock();
	int getBlockType();
	int getDeviceNumber();

	bool isProcessingUnitCPU();
	bool isProcessingUnitGPU();

	double getStepError(double timestep);

	void confirmStep(double timestep);
	void rejectStep(double timestep);

	double* addNewBlockBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength);
	double* addNewExternalBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength, double* border);

	void moveTempBorderVectorToBorderArray();

	void loadData(double* data);
	void getCurrentState(double* result);

	void saveState(std::ofstream& out);
	void loadState(std::ifstream& in);

private:
	ProcessingUnit* pu;

	ProblemType* problem;

	double* mParams;

	int blockNumber;


	func_ptr_t* mUserFuncs;
	initfunc_fill_ptr_t* mUserInitFuncs;
	unsigned short int* mCompFuncNumber;
	unsigned short int* mInitFuncNumber;


	int* sendBorderInfo;
	std::vector<int> tempSendBorderInfo;

	int* receiveBorderInfo;
	std::vector<int> tempReceiveBorderInfo;


	double** blockBorder;
	std::vector<double*> tempBlockBorder;


	double** externalBorder;
	std::vector<double*> tempExternalBorder;


	int countSendSegmentBorder;
	int countReceiveSegmentBorder;


	double* getNewBlockBorder(Block* neighbor, int borderLength);
	double* getNewExternalBorder(Block* neighbor, int borderLength, double* border);

	ProblemType* createProblem(int problemType, int solverType, double aTol, double rTol);
};

#endif /* SRC_BLOCKS_REALBLOCK_H_ */
