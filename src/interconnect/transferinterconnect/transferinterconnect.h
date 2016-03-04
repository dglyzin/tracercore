/*
 * transferinterconnect.h
 *
 *  Created on: 19 февр. 2016 г.
 *      Author: frolov
 */

#ifndef SRC_INTERCONNECT_TRANSFERINTERCONNECT_H_
#define SRC_INTERCONNECT_TRANSFERINTERCONNECT_H_

#include <mpi.h>

#include "../interconnect.h"

class TransferInterconnect: public Interconnect {
public:
	TransferInterconnect(int _sourceLocationNode, int _destinationLocationNode, int _borderLength,
			MPI_Comm* _pworkerComm);
	virtual ~TransferInterconnect();

	void wait();

protected:
	MPI_Comm* mpWorkerComm;

	int borderLength;

	MPI_Status* status;
	MPI_Request* request;
};

#endif /* SRC_INTERCONNECT_TRANSFERINTERCONNECT_H_ */
