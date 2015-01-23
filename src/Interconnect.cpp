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
		int _lengthBorder,
		double* _sourceBlockBorder, double* _destinationExternalBorder) {
	sourceLocationNode = _sourceLocationNode;
	destinationLocationNode = _destinationLocationNode;

	sourceType = _sourceType;
	destinationType = _destinationType;

	lengthBorder = _lengthBorder;

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
		MPI_Isend(sourceBlockBorder, lengthBorder, MPI_DOUBLE, destinationLocationNode, 999, MPI_COMM_WORLD, &request);
		MPI_Recv(destinationExternalBorder, lengthBorder, MPI_DOUBLE, sourceLocationNode, 999, MPI_COMM_WORLD, &status);
		return;
	}

	if(locationNode == sourceLocationNode)
	{
		MPI_Send(sourceBlockBorder, lengthBorder, MPI_DOUBLE, destinationLocationNode, 999, MPI_COMM_WORLD);
		return;
	}

	if(locationNode == destinationLocationNode) {
		MPI_Recv(destinationExternalBorder, lengthBorder, MPI_DOUBLE, sourceLocationNode, 999, MPI_COMM_WORLD, &status);
		return;
	}
}
