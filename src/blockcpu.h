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
	BlockCpu(int _length, int _width, int _lengthMove, int _widthMove, int _nodeNumber, int _deviceNumber);

	~BlockCpu();

	bool isRealBlock() { return true; }

	void prepareData();

	void computeOneStep(double dX2, double dY2, double dT);
	void computeOneStepBorder(double dX2, double dY2, double dT);
	void computeOneStepCenter(double dX2, double dY2, double dT);

	int getBlockType() { return CPU; }

	double* getCurrentState();

	void print();

	double* addNewBlockBorder(Block* neighbor, int side, int move, int borderLength);
	double* addNewExternalBorder(Block* neighbor, int side, int move, int borderLength, double* border);

	void moveTempBorderVectorToBorderArray();

	void loadData(double* data);
};

#endif /* SRC_BLOCKCPU_H_ */
