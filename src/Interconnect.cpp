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
		int _lengthBoundary,
		double* _sourceBlockBoundary, double* _destinationExternalBoundary) {
	sourceLocationNode = _sourceLocationNode;
	destinationLocationNode = _destinationLocationNode;

	sourceType = _sourceType;
	destinationType = _destinationType;

	lengthBoundary = _lengthBoundary;

	sourceBlockBoundary = _sourceBlockBoundary;
	destinationExternalBoundary = _destinationExternalBoundary;
}

Interconnect::~Interconnect() {
	// TODO Auto-generated destructor stub
}

void Interconnect::sendRecv(int locationNode) {
	printf("\nnode %d, source %d, dest %d\n", locationNode, sourceLocationNode, destinationLocationNode);
	if(locationNode == sourceLocationNode && locationNode == destinationLocationNode)
		return;

	if(locationNode == sourceLocationNode)
	{
		MPI_Send(sourceBlockBoundary, lengthBoundary, MPI_DOUBLE, destinationLocationNode, 999, MPI_COMM_WORLD);
		//printf("\nSEND_RECV comment/ Don't real working!!!\n\n");
		return;
	}

	if(locationNode == destinationLocationNode) {
		MPI_Recv(destinationExternalBoundary, lengthBoundary, MPI_DOUBLE, sourceLocationNode, 999, MPI_COMM_WORLD, &status);
		//printf("\nSEND_RECV comment/ Don't real working!!!\n\n");
		return;
	}
}
