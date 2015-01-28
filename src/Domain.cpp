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
}

Domain::~Domain() {
	// TODO Auto-generated destructor stub
}

void Domain::count() {
	double dX = 1./areaWidth;
	double dY = 1./areaLength;

	double dX2 = dX * dX;
	double dY2 = dY * dY;

	double dT = ( dX2 * dY2 ) / ( 2 * ( dX2 + dY2 ) );

	int repeatCount = (int)(1 / dT) + 1;

	for (int i = 0; i < repeatCount; ++i)
		nextStep(dX2, dY2, dT);
}

void Domain::nextStep(double dX2, double dY2, double dT) {
	for (int i = 0; i < blockCount; ++i)
		mBlocks[i]->prepareData();

	for (int i = 0; i < connectionCount; ++i)
		mInterconnects[i]->sendRecv(world_rank);

	for (int i = 0; i < blockCount; ++i)
		mBlocks[i]->courted(dX2, dY2, dT);
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
			for (int j = 0; j < widthArea; ++j)
				fprintf(out, "%d %d %f\n", i, j, resaultAll[i][j]);
			fprintf(out, "\n");
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

	in >> connectionCount;

	mInterconnects = new Interconnect* [connectionCount];

	for (int i = 0; i < connectionCount; ++i)
		mInterconnects[i] = readConnection(in);
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

	if(mBlocks[destination]->isRealBlock())
		for (int i = 0; i < borderLength; ++i)
			borderType[i + connectionDestinationMove] = BY_ANOTHER_BLOCK;

	return new Interconnect(sourceNode, destinationNode, sourceType, destionationType, borderLength, sourceData, destinationData);
}

int Domain::getCountGridNodes() {
	int count = 0;
	for (int i = 0; i < blockCount; ++i)
		count += mBlocks[i]->getCountGridNodes();

	return count;
}

