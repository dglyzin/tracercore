/*
 * nontransferinterconnect.cpp
 *
 *  Created on: 19 февр. 2016 г.
 *      Author: frolov
 */

#include "nontransferinterconnect.h"

NonTransferInterconnect::NonTransferInterconnect(int _sourceLocationNode, int _destinationLocationNode) :
		Interconnect(_sourceLocationNode, _destinationLocationNode) {
}

NonTransferInterconnect::~NonTransferInterconnect() {
}

void NonTransferInterconnect::wait() {
	return;
}

void NonTransferInterconnect::transfer() {
	return;
}

void NonTransferInterconnect::printTypeInformation() {
	printf("Non transfer interconnect\n");
}

void NonTransferInterconnect::printMemoryAddresInformation() {
	return;
}
