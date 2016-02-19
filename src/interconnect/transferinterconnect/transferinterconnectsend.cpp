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

