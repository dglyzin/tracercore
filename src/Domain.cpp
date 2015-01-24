/*
 * Domain.cpp
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#include "Domain.h"

using namespace std;

Domain::Domain(int world_rank, int world_size, int blockCount, int borderCount) {
	setDefaultValue();

	int countForThread = blockCount / world_size;
	mBlocks = new Block* [countForThread];
	mInterconnects = new Interconnect* [borderCount*2];

	for (int i = 0; i < blockCount; ++i)
		mBlocks[i] = new BlockNull(world_rank);

	for (int i = countForThread*world_rank; i < countForThread*(world_rank + 1); ++i) {
		delete mBlocks[i];
		mBlocks[i] = new BlockCpu(blockLengthSize[i], blockWidthSize[i], world_rank);
	}

	// 1 - 2 node 0/1
	mInterconnects[0] = new Interconnect(0, 1, CPU, CPU, b1_b2_border_length, mBlocks[1]->getTopBlockBorder() + b1_top_border_move, mBlocks[2]->getBottomExternalBorder() + b2_bottom_border_move);
	// 2 - 1 node 1/0
	mInterconnects[1] = new Interconnect(1, 0, CPU, CPU, b1_b2_border_length, mBlocks[2]->getBottomBlockBorder() + b2_bottom_border_move, mBlocks[1]->getTopExternalBorder() + b1_top_border_move);
	// 0 - 1 node 0/0
	mInterconnects[2] = new Interconnect(0, 0, CPU, CPU, b0_b1_border_length, mBlocks[0]->getRightBlockBorder() + b0_right_border_move, mBlocks[1]->getLeftExternalBorder() + b1_left_border_move);
	// 1 - 0 node 0/0
	mInterconnects[3] = new Interconnect(0, 0, CPU, CPU, b0_b1_border_length, mBlocks[1]->getLeftBlockBorder() + b1_left_border_move, mBlocks[0]->getRightExternalBorder() + b0_right_border_move);
	// 1 - 3 node 0/1
	mInterconnects[4] = new Interconnect(0, 1, CPU, CPU, b1_b3_border_length, mBlocks[1]->getRightBlockBorder() + b1_right_border_move, mBlocks[3]->getLeftExternalBorder() + b3_left_border_move);
	// 3 - 1 node 1/0
	mInterconnects[5] = new Interconnect(1, 0, CPU, CPU, b1_b3_border_length, mBlocks[3]->getLeftBlockBorder() + b3_left_border_move, mBlocks[1]->getRightExternalBorder() + b1_right_border_move);

	if( mBlocks[0]->isRealBlock() ) {
		Block* b = mBlocks[0];

		int* topBorderType = b->getTopBorderType();
		int* leftBorderType = b->getLeftBorderType();
		int* bottomBorderType = b->getBottomBorderType();
		int* rightBorderType = b->getRightBorderType();

		for (int i = 0; i < b0_width; ++i)
			topBorderType[i] = BY_FUNCTION;

		for (int i = 0; i < b0_length; ++i)
			leftBorderType[i] = BY_FUNCTION;

		for (int i = 0; i < b0_width; ++i)
			bottomBorderType[i] = BY_FUNCTION;

		for (int i = 0; i < b0_right_border_move; ++i)
			rightBorderType[i] = BY_FUNCTION;
		for (int i = b0_right_border_move; i < b0_b1_border_length + b0_right_border_move; ++i)
			rightBorderType[i] = BY_ANOTHER_BLOCK;
		for (int i = b0_b1_border_length + b0_right_border_move; i < b0_length; ++i)
			rightBorderType[i] = BY_FUNCTION;


		double* topBlockBorder = b->getTopBlockBorder();
		double* leftBlockBorder = b->getLeftBlockBorder();
		double* bottomBlockBorder = b->getBottomBlockBorder();
		double* rightBlockBorder = b->getRightBlockBorder();

		double* topExternalBorder = b->getTopExternalBorder();
		double* leftExternalBorder = b->getLeftExternalBorder();
		double* bottomExternalBorder = b->getBottomExternalBorder();
		double* rightExternalBorder = b->getRightExternalBorder();

		for (int i = 0; i < b0_width; ++i) {
			topBlockBorder[i] = 0;
			bottomBlockBorder[i] = 0;

			topExternalBorder[i] = 100 * cos( (i - 12) / 9. );
			bottomExternalBorder[i] = 10;
		}

		for (int i = 0; i < b0_length; ++i) {
			leftBlockBorder[i] = 0;
			rightBlockBorder[i] = 0;

			leftExternalBorder[i] = 10;
			rightExternalBorder[i] = 10;
		}
	}

	if( mBlocks[1]->isRealBlock() ) {
		Block* b = mBlocks[1];

		int* topBorderType = b->getTopBorderType();
		int* leftBorderType = b->getLeftBorderType();
		int* bottomBorderType = b->getBottomBorderType();
		int* rightBorderType = b->getRightBorderType();


		for (int i = 0; i < b1_top_border_move; ++i)
			topBorderType[i] = BY_FUNCTION;
		for (int i = b1_top_border_move; i < b1_top_border_move + b1_b2_border_length; ++i)
			topBorderType[i] = BY_ANOTHER_BLOCK;
		for (int i = b1_top_border_move + b1_b2_border_length; i < b1_width; ++i)
			topBorderType[i] = BY_FUNCTION;

		for (int i = 0; i < b1_length; ++i)
			leftBorderType[i] = BY_ANOTHER_BLOCK;

		for (int i = 0; i < b1_width; ++i)
			bottomBorderType[i] = BY_FUNCTION;

		for (int i = 0; i < b1_length; ++i)
			rightBorderType[i] = BY_ANOTHER_BLOCK;


		double* topBlockBorder = b->getTopBlockBorder();
		double* leftBlockBorder = b->getLeftBlockBorder();
		double* bottomBlockBorder = b->getBottomBlockBorder();
		double* rightBlockBorder = b->getRightBlockBorder();

		double* topExternalBorder = b->getTopExternalBorder();
		double* leftExternalBorder = b->getLeftExternalBorder();
		double* bottomExternalBorder = b->getBottomExternalBorder();
		double* rightExternalBorder = b->getRightExternalBorder();

		for (int i = 0; i < b1_width; ++i) {
			topBlockBorder[i] = 0;
			bottomBlockBorder[i] = 0;

			topExternalBorder[i] = 100 * cos( (i - 25) / 20. );
			bottomExternalBorder[i] = 10;
		}

		for (int i = 0; i < b1_length; ++i) {
			leftBlockBorder[i] = 0;
			rightBlockBorder[i] = 0;

			leftExternalBorder[i] = 10;
			rightExternalBorder[i] = 10;
		}
	}

	if( mBlocks[2]->isRealBlock() ) {
		Block* b = mBlocks[2];

		int* topBorderType = b->getTopBorderType();
		int* leftBorderType = b->getLeftBorderType();
		int* bottomBorderType = b->getBottomBorderType();
		int* rightBorderType = b->getRightBorderType();

		for (int i = 0; i < b2_width; ++i)
			topBorderType[i] = BY_FUNCTION;

		for (int i = 0; i < b2_length; ++i)
			leftBorderType[i] = BY_FUNCTION;

		for (int i = 0; i < b2_width; ++i)
			bottomBorderType[i] = BY_ANOTHER_BLOCK;

		for (int i = 0; i < b2_length; ++i)
			rightBorderType[i] = BY_FUNCTION;


		double* topBlockBorder = b->getTopBlockBorder();
		double* leftBlockBorder = b->getLeftBlockBorder();
		double* bottomBlockBorder = b->getBottomBlockBorder();
		double* rightBlockBorder = b->getRightBlockBorder();

		double* topExternalBorder = b->getTopExternalBorder();
		double* leftExternalBorder = b->getLeftExternalBorder();
		double* bottomExternalBorder = b->getBottomExternalBorder();
		double* rightExternalBorder = b->getRightExternalBorder();

		for (int i = 0; i < b2_width; ++i) {
			topBlockBorder[i] = 0;
			bottomBlockBorder[i] = 0;

			topExternalBorder[i] = 100 * cos( (i - 12) / 9. );
			bottomExternalBorder[i] = 100 * cos( (i - 25) / 20. );
		}

		for (int i = 0; i < b2_length; ++i) {
			leftBlockBorder[i] = 0;
			rightBlockBorder[i] = 0;

			leftExternalBorder[i] = 10;
			rightExternalBorder[i] = 10;
		}
	}

	if( mBlocks[3]->isRealBlock() ) {
		Block* b = mBlocks[3];

		int* topBorderType = b->getTopBorderType();
		int* leftBorderType = b->getLeftBorderType();
		int* bottomBorderType = b->getBottomBorderType();
		int* rightBorderType = b->getRightBorderType();

		for (int i = 0; i < b3_width; ++i)
			topBorderType[i] = BY_FUNCTION;

		for (int i = 0; i < b3_left_border_move; ++i)
			leftBorderType[i] = BY_FUNCTION;
		for (int i = b3_left_border_move; i < b3_left_border_move + b1_b3_border_length; ++i)
			leftBorderType[i] = BY_ANOTHER_BLOCK;
		for (int i = b3_left_border_move + b1_b3_border_length; i < b3_length; ++i)
			leftBorderType[i] = BY_FUNCTION;

		for (int i = 0; i < b3_width; ++i)
			bottomBorderType[i] = BY_FUNCTION;

		for (int i = 0; i < b3_length; ++i)
			rightBorderType[i] = BY_FUNCTION;


		double* topBlockBorder = b->getTopBlockBorder();
		double* leftBlockBorder = b->getLeftBlockBorder();
		double* bottomBlockBorder = b->getBottomBlockBorder();
		double* rightBlockBorder = b->getRightBlockBorder();

		double* topExternalBorder = b->getTopExternalBorder();
		double* leftExternalBorder = b->getLeftExternalBorder();
		double* bottomExternalBorder = b->getBottomExternalBorder();
		double* rightExternalBorder = b->getRightExternalBorder();

		for (int i = 0; i < b3_width; ++i) {
			topBlockBorder[i] = 0;
			bottomBlockBorder[i] = 0;

			topExternalBorder[i] = 100 * cos( (i - 12) / 9. );
			bottomExternalBorder[i] = 10;
		}

		for (int i = 0; i < b3_length; ++i) {
			leftBlockBorder[i] = 0;
			rightBlockBorder[i] = 0;

			leftExternalBorder[i] = 10;
			rightExternalBorder[i] = 10;
		}
	}
}

Domain::~Domain() {
	// TODO Auto-generated destructor stub
}

void Domain::calc(int world_rank, int blockCount, int borderCount) {
	for (int i = 0; i < blockCount; ++i)
		if( mBlocks[i]->isRealBlock() ) {
			mBlocks[i]->courted();
			mBlocks[i]->prepareData();
		}

	for (int i = 0; i < borderCount*2; ++i)
		mInterconnects[i]->sendRecv(world_rank);
}

void Domain::print(int world_rank, int blockCount) {
	if(world_rank == 0) {
		double** resaultAll = new double* [85];
		for (int i = 0; i < 85; ++i)
			resaultAll[i] = new double[100];

		for (int i = 0; i < 85; ++i)
			for (int j = 0; j < 100; ++j)
				resaultAll[i][j] = 0;

		for (int i = 0; i < blockCount; ++i) {
			if(mBlocks[i]->isRealBlock()) {
				double** resault = mBlocks[i]->getResault();

				for (int j = 0; j < blockLengthSize[i]; ++j)
					for (int k = 0; k < blockWidthSize[i]; ++k)
						resaultAll[j + blockMoveLenght[i]][k + blockMoveWidth[i]] = resault[j][k];
			}
			else
				for (int j = 0; j < blockLengthSize[i]; ++j)
					MPI_Recv(resaultAll[j + blockMoveLenght[i]] + blockMoveWidth[i], blockWidthSize[i], MPI_DOUBLE, i/2, 999, MPI_COMM_WORLD, &status);
		}

		FILE* out = fopen("res", "wb");

		for (int i = 0; i < 85; ++i) {
			for (int j = 0; j < 100; ++j)
				fprintf(out, "%d %d %f\n", i, j, resaultAll[i][j]);
			fprintf(out, "\n");
		}

		fclose(out);
	}
	else {
		for (int i = 0; i < blockCount; ++i) {
			if(mBlocks[i]->isRealBlock()) {
				double** resault = mBlocks[i]->getResault();

				for (int j = 0; j < blockLengthSize[i]; ++j)
					MPI_Send(resault[j], blockWidthSize[i], MPI_DOUBLE, 0, 999, MPI_COMM_WORLD);
			}
		}
	}
}

void Domain::readFromFile(string path) {
}

void Domain::readLengthAndWidthArea(ifstream in) {

}

void Domain::setDefaultValue() {
	blockLengthSize[0] = b0_length;
	blockLengthSize[1] = b1_length;
	blockLengthSize[2] = b2_length;
	blockLengthSize[3] = b3_length;

	blockWidthSize[0] = b0_width;
	blockWidthSize[1] = b1_width;
	blockWidthSize[2] = b2_width;
	blockWidthSize[3] = b3_width;

	blockMoveLenght[0] = b0_moveL;
	blockMoveLenght[1] = b1_moveL;
	blockMoveLenght[2] = b2_moveL;
	blockMoveLenght[3] = b3_moveL;

	blockMoveWidth[0] = b0_moveW;
	blockMoveWidth[1] = b1_moveW;
	blockMoveWidth[2] = b2_moveW;
	blockMoveWidth[3] = b3_moveW;
}

