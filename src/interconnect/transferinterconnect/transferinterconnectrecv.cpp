/*
 * transferinterconnectrecv.cpp
 *
 *  Created on: 19 февр. 2016 г.
 *      Author: frolov
 */

#include "transferinterconnectrecv.h"

TransferInterconnectRecv::TransferInterconnectRecv(int _sourceLocationNode,
		int _destinationLocationNode, int _borderLength,
		double* _destinationExternalBorder, MPI_Comm* _pworkerComm) :
		TransferInterconnect(_sourceLocationNode, _destinationLocationNode,
				_borderLength, _pworkerComm) {
	destinationExternalBorder = _destinationExternalBorder;
}

TransferInterconnectRecv::~TransferInterconnectRecv() {
}

