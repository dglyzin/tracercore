/*
 * realblock.h
 *
 *  Created on: 15 окт. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKS_REALBLOCK_H_
#define SRC_BLOCKS_REALBLOCK_H_

#include <vector>

#include "../processingunit/processingunit.h"

#include "../utils.h"
#include "block.h"

#include "../numericalmethod/numericalmethod.h"
#include "../problem/problem.h"
#include "../state.h"

class RealBlock: public Block {
public:
	RealBlock(int _nodeNumber, int _dimension, int _xCount, int _yCount, int _zCount, int _cellSize, int _haloSize,
			int _blockNumber, ProcessingUnit* _pu, unsigned short int* _initFuncNumber,
			unsigned short int* _compFuncNumber, Problem* _problem, NumericalMethod* _method);

	virtual ~RealBlock();

	//void afterCreate(int problemType, int solverType, double aTol, double rTol);

	void computeStageBorder(int stage, double time);
	void computeStageCenter(int stage, double time);

	void prepareArgument(int stage, double timeStep);
	void getSubVolume(double* result, int mStart, int mStop, int nStart, int nStop, int side);
	void setSubVolume(double* source, int mStart, int mStop, int nStart, int nStop, int side);

	void prepareStageData(int stage);
	void prepareStageSourceResult(int stage, double timeStep, double currentTime);

	bool isRealBlock();
	int getBlockType();
	int getDeviceNumber();

	bool isBlockType(int type);
	bool isDeviceNumber(int number);

	bool isProcessingUnitCPU();
	bool isProcessingUnitGPU();

	double getStepError(double timeStep);

	void confirmStep(double timeStep);
	void rejectStep(double timeStep);

	double* addNewBlockBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength);
	double* addNewExternalBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength,
			double* border);

	void moveTempBorderVectorToBorderArray();

	void getCurrentState(double* result);

	void saveStateForDraw(char* path);
	void saveStateForLoad(char* path);
	void saveStateForDrawDenseOutput(char* path, double timeStep, double tetha);
	void loadState(std::ifstream& in);

	bool isNan();
	ProcessingUnit* getPU();

	void print();

private:
	ProcessingUnit* pu;

	State** mStates;
	Problem* mProblem;
	NumericalMethod* mNumericalMethod;

	double** mSource;
	double* mResult;

	double** mDelayArrays;

	double** mCommonTempStorages;

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

	void printGeneralInformation();
	void printBorderInfo();
	void printData();
};

#endif /* SRC_BLOCKS_REALBLOCK_H_ */
