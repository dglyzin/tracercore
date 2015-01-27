/*
 * Domain.cpp
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#include "Domain.h"

using namespace std;

Domain::Domain(int _world_rank, int _world_size, char* path) {
	world_rank = _world_rank;
	world_size = _world_size;

	readFromFile(path);
	printf("\n************\n");

	/*blockCount = _blockCount;
	connectionCount = _connectionCount;*/

	//setDefaultValue();

	//int countForThread = blockCount / world_size;
	//mBlocks = new Block* [blockCount];
	//mInterconnects = new Interconnect* [connectionCount];

	/*for (int i = 0; i < blockCount; ++i)
		mBlocks[i] = new BlockNull(blockLengthSize[i], blockWidthSize[i], blockMoveLenght[i], blockMoveWidth[i], world_rank);

	for (int i = countForThread*world_rank; i < countForThread*(world_rank + 1); ++i) {
		delete mBlocks[i];
		mBlocks[i] = new BlockCpu(blockLengthSize[i], blockWidthSize[i], blockMoveLenght[i], blockMoveWidth[i], world_rank);
	}*/

	/*for (int i = 0; i < blockCount; ++i)
		if( blockThread[i] == world_rank )
			mBlocks[i] = new BlockCpu(blockLengthSize[i], blockWidthSize[i], blockMoveLenght[i], blockMoveWidth[i], blockThread[i]);
		else
			mBlocks[i] = new BlockNull(blockLengthSize[i], blockWidthSize[i], blockMoveLenght[i], blockMoveWidth[i], blockThread[i]);*/
/*
	// 1 - 2 node 0/1
	mInterconnects[0] = new Interconnect(mBlocks[1]->getWorldRank(), mBlocks[2]->getWorldRank(), CPU, CPU,
			b1_b2_border_length, mBlocks[1]->getTopBlockBorder() + b1_top_border_move, mBlocks[2]->getBottomExternalBorder() + b2_bottom_border_move);
	// 2 - 1 node 1/0
	mInterconnects[1] = new Interconnect(mBlocks[2]->getWorldRank(), mBlocks[1]->getWorldRank(), CPU, CPU,
			b1_b2_border_length, mBlocks[2]->getBottomBlockBorder() + b2_bottom_border_move, mBlocks[1]->getTopExternalBorder() + b1_top_border_move);
	// 0 - 1 node 0/0
	mInterconnects[2] = new Interconnect(mBlocks[0]->getWorldRank(), mBlocks[1]->getWorldRank(), CPU, CPU,
			b0_b1_border_length, mBlocks[0]->getRightBlockBorder() + b0_right_border_move, mBlocks[1]->getLeftExternalBorder() + b1_left_border_move);
	// 1 - 0 node 0/0
	mInterconnects[3] = new Interconnect(mBlocks[1]->getWorldRank(), mBlocks[0]->getWorldRank(), CPU, CPU,
			b0_b1_border_length, mBlocks[1]->getLeftBlockBorder() + b1_left_border_move, mBlocks[0]->getRightExternalBorder() + b0_right_border_move);
	// 1 - 3 node 0/1
	mInterconnects[4] = new Interconnect(mBlocks[1]->getWorldRank(), mBlocks[3]->getWorldRank(), CPU, CPU,
			b1_b3_border_length, mBlocks[1]->getRightBlockBorder() + b1_right_border_move, mBlocks[3]->getLeftExternalBorder() + b3_left_border_move);
	// 3 - 1 node 1/0
	mInterconnects[5] = new Interconnect(mBlocks[3]->getWorldRank(), mBlocks[1]->getWorldRank(), CPU, CPU,
			b1_b3_border_length, mBlocks[3]->getLeftBlockBorder() + b3_left_border_move, mBlocks[1]->getRightExternalBorder() + b1_right_border_move);
			*/

	/*if( mBlocks[0]->isRealBlock() ) {
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
	}*/
}

Domain::~Domain() {
	// TODO Auto-generated destructor stub
}

void Domain::calc() {
	for (int i = 0; i < blockCount; ++i)
		mBlocks[i]->prepareData();

	for (int i = 0; i < connectionCount; ++i)
		mInterconnects[i]->sendRecv(world_rank);

	for (int i = 0; i < blockCount; ++i)
		mBlocks[i]->courted(1./areaWidth, 1./areaLength);

	char c;

	/*if(world_rank == 0) {
		for (int i = 0; i < blockCount; ++i)
			mBlocks[i]->print(world_rank);

		scanf("%c", &c);
	}
	else
		scanf("%c", &c);

	if(world_rank == 1) {
		for (int i = 0; i < blockCount; ++i)
			mBlocks[i]->print(world_rank);

		scanf("%c", &c);
	}
	else
		scanf("%c", &c);

	if(world_rank == 2) {
		for (int i = 0; i < blockCount; ++i)
			mBlocks[i]->print(world_rank);

		scanf("%c", &c);
	}
	else
		scanf("%c", &c);

	if(world_rank == 3) {
		for (int i = 0; i < blockCount; ++i)
			mBlocks[i]->print(world_rank);

		scanf("%c", &c);
	}
	else
		scanf("%c", &c);*/
	//print("");
	//scanf("%c", &c);
}

void Domain::print(char* path) {
	if(world_rank == 0) {
		double** resaultAll = new double* [lengthArea];
		for (int i = 0; i < lengthArea; ++i)
			resaultAll[i] = new double[widthArea];

		for (int i = 0; i < lengthArea; ++i)
			for (int j = 0; j < widthArea; ++j)
				resaultAll[i][j] = 0;

		for (int i = 0; i < blockCount; ++i) {
			if(mBlocks[i]->isRealBlock()) {
				double** resault = mBlocks[i]->getResault();

				for (int j = 0; j < mBlocks[i]->getLength(); ++j)
					for (int k = 0; k < mBlocks[i]->getWidth(); ++k)
						resaultAll[j + mBlocks[i]->getLenghtMove()][k + mBlocks[i]->getWidthMove()] = resault[j][k];
			}
			else
				for (int j = 0; j < mBlocks[i]->getLength(); ++j)
					MPI_Recv(resaultAll[j + mBlocks[i]->getLenghtMove()] + mBlocks[i]->getWidthMove(), mBlocks[i]->getWidth(), MPI_DOUBLE, mBlocks[i]->getNodeNumber(), 999, MPI_COMM_WORLD, &status);
		}

		FILE* out = fopen(path, "wb");

		for (int i = 0; i < lengthArea; ++i) {
			for (int j = 0; j < widthArea; ++j) {
				fprintf(out, "%d %d %f\n", i, j, resaultAll[i][j]);
				/*if( j % (widthArea / world_size) == 0)
					printf("\t");
				printf("%6.1f", resaultAll[i][j]);*/
			}
			fprintf(out, "\n");
			//printf("\n");
		}

		fclose(out);
	}
	else {
		for (int i = 0; i < blockCount; ++i) {
			if(mBlocks[i]->isRealBlock()) {
				double** resault = mBlocks[i]->getResault();

				for (int j = 0; j < mBlocks[i]->getLength(); ++j)
					MPI_Send(resault[j], mBlocks[i]->getWidth(), MPI_DOUBLE, 0, 999, MPI_COMM_WORLD);
			}
		}
	}
}

void Domain::readFromFile(char* path) {
	ifstream in;
	in.open(path);

	readLengthAndWidthArea(in);
	in >> blockCount;

	mBlocks = new Block* [blockCount];

	for (int i = 0; i < blockCount; ++i)
		mBlocks[i] = readBlock(in);

	/*if(world_rank == 0)
	for (int i = 0; i < blockCount; ++i)
		mBlocks[i]->print(world_rank);*/

	in >> connectionCount;

	mInterconnects = new Interconnect* [connectionCount];

	for (int i = 0; i < connectionCount; ++i)
		mInterconnects[i] = readConnection(in);

	/*if(world_rank == 1)
	for (int i = 0; i < connectionCount; ++i)
		mInterconnects[i]->print(world_rank);

	if(world_rank == 1)
	for (int i = 0; i < blockCount; ++i)
		mBlocks[i]->print(world_rank);*/
}

void Domain::readLengthAndWidthArea(ifstream& in) {
	in >> lengthArea;
	in >> widthArea;
}

Block* Domain::readBlock(ifstream& in) {
	int length;
	int width;

	int lengthMove;
	int widthMove;

	int word_rank_creator;

	in >> length;
	in >> width;

	in >> lengthMove;
	in >> widthMove;

	in >> word_rank_creator;

	// TODO добавить нужный конструктор классам. реализовать иную схему получения блоков
	if(word_rank_creator == world_rank)
		return new BlockCpu(length, width, lengthMove, widthMove, word_rank_creator);
	else
		return new BlockNull(length, width, lengthMove, widthMove, word_rank_creator);
}

Interconnect* Domain::readConnection(ifstream& in) {
	int source;
	int destination;

	char borderSide;

	int connectionSourceMove;
	int connectionDestinationMove;
	int borderLength;

	in >> source;
	in >> destination;

	in >> borderSide;

	in >> connectionSourceMove;
	in >> connectionDestinationMove;

	in >> borderLength;

	int* borderType;

	int sourceNode = mBlocks[source]->getNodeNumber();
	int destinationNode = mBlocks[destination]->getNodeNumber();

	int sourceType = mBlocks[source]->getBlockType();
	int destionationType = mBlocks[destination]->getBlockType();

	double* sourceData;
	double* destinationData;

	//printf("\nsource %d, dest %d, borderSide %c, sMove %d, dMove %d, bL %d\n", source, destination, borderSide, connectionSourceMove, connectionDestinationMove, borderLength);

	switch (borderSide) {
		case 't':
			borderType = mBlocks[destination]->getTopBorderType();

			sourceData = mBlocks[source]->getBottomBlockBorder() + connectionSourceMove;
			destinationData = mBlocks[destination]->getTopExternalBorder() + connectionDestinationMove;

			break;

		case 'l':
			borderType = mBlocks[destination]->getLeftBorderType();

			sourceData = mBlocks[source]->getRightBlockBorder() + connectionSourceMove;
			destinationData = mBlocks[destination]->getLeftExternalBorder() + connectionDestinationMove;

			break;

		case 'b':
			borderType = mBlocks[destination]->getBottomBorderType();

			sourceData = mBlocks[source]->getTopBlockBorder() + connectionSourceMove;
			destinationData = mBlocks[destination]->getBottomExternalBorder() + connectionDestinationMove;

			break;

		case 'r':
			borderType = mBlocks[destination]->getRightBorderType();

			sourceData = mBlocks[source]->getLeftBlockBorder() + connectionSourceMove;
			destinationData = mBlocks[destination]->getRightExternalBorder() + connectionDestinationMove;

			break;

		default:
			// TODO Рассматривать случай?
			return NULL;
	}

	// TODO maybe isRealBlock?
	if(world_rank == mBlocks[destination]->getNodeNumber())
		for (int i = 0; i < borderLength; ++i)
			borderType[i + connectionDestinationMove] = BY_ANOTHER_BLOCK;

	return new Interconnect(sourceNode, destinationNode, sourceType, destionationType, borderLength, sourceData, destinationData);
}

void Domain::setDefaultValue() {
	/*blockLengthSize[0] = b0_length;
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

	blockThread[0] = 1;
	blockThread[1] = 1;
	blockThread[2] = 0;
	blockThread[3] = 0;*/
}

