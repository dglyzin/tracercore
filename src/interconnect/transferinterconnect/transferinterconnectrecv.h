/*
 * transferinterconnectrecv.h
 *
 *  Created on: 19 февр. 2016 г.
 *      Author: frolov
 */

#ifndef SRC_INTERCONNECT_TRANSFERINTERCONNECT_TRANSFERINTERCONNECTRECV_H_
#define SRC_INTERCONNECT_TRANSFERINTERCONNECT_TRANSFERINTERCONNECTRECV_H_

#include "transferinterconnect.h"

class TransferInterconnectRecv: public TransferInterconnect {
public:
	TransferInterconnectRecv(int _sourceLocationNode, int _destinationLocationNode, int _borderLength,
			double* _destinationExternalBorder, MPI_Comm* _pworkerComm);
	virtual ~TransferInterconnectRecv();

	void transfer();

private:
	double* destinationExternalBorder;
};

#endif /* SRC_INTERCONNECT_TRANSFERINTERCONNECT_TRANSFERINTERCONNECTRECV_H_ */
