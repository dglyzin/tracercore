/*
 * Interconnect.cpp
 *
 *  Created on: 19 янв. 2015 г.
 *      Author: frolov
 */

#include "Interconnect.h"

using namespace std;

Interconnect::Interconnect(int _sourceLocationNode, int _destinationLocationNode,
		int _sourceType, int _destinationType,
		int _borderLength,
		double* _sourceBlockBorder, double* _destinationExternalBorder) {
	sourceLocationNode = _sourceLocationNode;
	destinationLocationNode = _destinationLocationNode;

	sourceType = _sourceType;
	destinationType = _destinationType;

	borderLength = _borderLength;

	sourceBlockBorder = _sourceBlockBorder;
	destinationExternalBorder = _destinationExternalBorder;
}

Interconnect::~Interconnect() {
	// TODO Auto-generated destructor stub
}

void Interconnect::sendRecv(int locationNode) {
	//printf("\nnode %d, source %d, dest %d\n", locationNode, sourceLocationNode, destinationLocationNode);
	/*
	 * TODO
	 * Пересылка для видеокарт и центральным процессоров.
	 * Пересылка для блоков внутри одного потока
	 */
	if(locationNode == sourceLocationNode && locationNode == destinationLocationNode) {
		MPI_Isend(sourceBlockBorder, borderLength, MPI_DOUBLE, destinationLocationNode, 999, MPI_COMM_WORLD, &request);
		MPI_Recv(destinationExternalBorder, borderLength, MPI_DOUBLE, sourceLocationNode, 999, MPI_COMM_WORLD, &status);
		return;
	}

	if(locationNode == sourceLocationNode)
	{
		MPI_Send(sourceBlockBorder, borderLength, MPI_DOUBLE, destinationLocationNode, 999, MPI_COMM_WORLD);
		return;
	}

	if(locationNode == destinationLocationNode) {
		MPI_Recv(destinationExternalBorder, borderLength, MPI_DOUBLE, sourceLocationNode, 999, MPI_COMM_WORLD, &status);
		return;
	}
}

void Interconnect::print(int locationNode) {
	printf("\nInterconnect\n");
	printf("\nnode %d, source %d, dest %d\n", locationNode, sourceLocationNode, destinationLocationNode);
}

int Interconnect::getDeviceNumber(int blockType) {
	switch (blockType) {
		case DEVICE0:
			return 0;
		case DEVICE1:
			return 1;
		case DEVICE2:
			return 2;
		default:
			return 0;
	}
}
