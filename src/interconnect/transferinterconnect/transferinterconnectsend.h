/*
 * transferinterconnectsend.h
 *
 *  Created on: 19 февр. 2016 г.
 *      Author: frolov
 */

#ifndef SRC_INTERCONNECT_TRANSFERINTERCONNECT_TRANSFERINTERCONNECTSEND_H_
#define SRC_INTERCONNECT_TRANSFERINTERCONNECT_TRANSFERINTERCONNECTSEND_H_

#include "transferinterconnect.h"

class TransferInterconnectSend: public TransferInterconnect {
public:
	TransferInterconnectSend(int _sourceLocationNode, int _destinationLocationNode, int _borderLength,
			double* _sourceBlockBorder, MPI_Comm* _pworkerComm);
	virtual ~TransferInterconnectSend();

	void transfer();

private:
	double* sourceBlockBorder;

	void printTypeInformation();
	void printMemoryAddresInformation();
};

#endif /* SRC_INTERCONNECT_TRANSFERINTERCONNECT_TRANSFERINTERCONNECTSEND_H_ */
