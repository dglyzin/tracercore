/*
 * transferinterconnectrecv.cpp
 *
 *  Created on: 19 февр. 2016 г.
 *      Author: frolov
 */

#include "transferinterconnectrecv.h"

TransferInterconnectRecv::TransferInterconnectRecv(int _sourceLocationNode, int _destinationLocationNode,
		int _borderLength, double* _destinationExternalBorder, MPI_Comm* _pworkerComm) :
		TransferInterconnect(_sourceLocationNode, _destinationLocationNode, _borderLength, _pworkerComm) {
	destinationExternalBorder = _destinationExternalBorder;
}

TransferInterconnectRecv::~TransferInterconnectRecv() {
}

void TransferInterconnectRecv::transfer() {
	MPI_Irecv(destinationExternalBorder, borderLength, MPI_DOUBLE, sourceLocationNode, 999, *mpWorkerComm, request);
	//MPI_Recv(destinationExternalBorder, borderLength, MPI_DOUBLE, sourceLocationNode, 999, *mpWorkerComm, status);
}

void TransferInterconnectRecv::printTypeInformation() {
	printf("Transfer receive interconnect\n");
}

void TransferInterconnectRecv::printMemoryAddresInformation() {
	printf("   Destination memory address: %p\n", destinationExternalBorder);
}
