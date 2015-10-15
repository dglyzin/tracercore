/*
 * realblock.h
 *
 *  Created on: 15 окт. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_BLOCKS_REALBLOCK_H_
#define SRC_BLOCKS_REALBLOCK_H_

#include <vector>

#include "block.h"

class RealBlock: public Block {
public:
	RealBlock();
	virtual ~RealBlock();

private:
	ProcessingUnit* pc;

	ProblemType* problem;


	func_ptr_t* mUserFuncs;
	initfunc_fill_ptr_t* mUserInitFuncs;
	unsigned short int* mCompFuncNumber;


	int* sendBorderInfo;
	std::vector<int> tempSendBorderInfo;

	int* receiveBorderInfo;
	std::vector<int> tempReceiveBorderInfo;


	double** blockBorder;
	int* blockBorderMemoryAllocType;
	std::vector<double*> tempBlockBorder;
	std::vector<int> tempBlockBorderMemoryAllocType;


	double** externalBorder;
	int* externalBorderMemoryAllocType;
	std::vector<double*> tempExternalBorder;
	std::vector<int> tempExternalBorderMemoryAllocType;


	int countSendSegmentBorder;
	int countReceiveSegmentBorder;
};

#endif /* SRC_BLOCKS_REALBLOCK_H_ */
