/*
 * transferinterconnectsend.cpp
 *
 *  Created on: 19 февр. 2016 г.
 *      Author: frolov
 */

#include "transferinterconnectsend.h"

TransferInterconnectSend::TransferInterconnectSend(int _sourceLocationNode,
		int _destinationLocationNode, int _borderLength,
		double* _sourceBlockBorder, MPI_Comm* _pworkerComm) :
		TransferInterconnect(_sourceLocationNode, _destinationLocationNode,
				_borderLength, _pworkerComm) {
	sourceBlockBorder = _sourceBlockBorder;
}

TransferInterconnectSend::~TransferInterconnectSend() {
}

void TransferInterconnectSend::transfer() {
	MPI_Isend(sourceBlockBorder, borderLength, MPI_DOUBLE, destinationLocationNode, 999, *mpWorkerComm, request);
	//MPI_Send(sourceBlockBorder, borderLength, MPI_DOUBLE, destinationLocationNode, 999, *mpWorkerComm);
}

void TransferInterconnectSend::printTypeInformation() {
	printf("Transfer send interconnect\n");
}

void TransferInterconnectSend::printMemoryAddresInformation() {
	printf("   Source memory address     : %p\n", sourceBlockBorder);
}
