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
	if(locationNode == sourceLocationNode && locationNode == destinationLocationNode)
		return;

	if(locationNode == sourceLocationNode)
	{
		MPI_Send(sourceBlockBorder, lengthBorder, MPI_DOUBLE, destinationLocationNode, 999, MPI_COMM_WORLD);
		//printf("\nSEND_RECV comment/ Don't real working!!!\n\n");
		return;
	}

	if(locationNode == destinationLocationNode) {
		MPI_Recv(destinationExternalBorder, lengthBorder, MPI_DOUBLE, sourceLocationNode, 999, MPI_COMM_WORLD, &status);
		//printf("\nSEND_RECV comment/ Don't real working!!!\n\n");
		return;
	}
}
