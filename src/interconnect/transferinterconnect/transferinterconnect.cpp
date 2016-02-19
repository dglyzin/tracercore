/*
 * transferinterconnect.cpp
 *
 *  Created on: 19 февр. 2016 г.
 *      Author: frolov
 */

#include "transferinterconnect.h"

TransferInterconnect::TransferInterconnect(int _sourceLocationNode,
		int _destinationLocationNode, int _borderLength, MPI_Comm* _pworkerComm) :
		Interconnect(_sourceLocationNode, _destinationLocationNode) {
	borderLength = _borderLength;

	request = new MPI_Request();

	status = new MPI_Status();

	mpWorkerComm = _pworkerComm;

}

TransferInterconnect::~TransferInterconnect() {
	if (request != NULL)
		delete request;

	if (status != NULL)
		delete status;
}

