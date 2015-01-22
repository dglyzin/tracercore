/*
 * Interconnect.h
 *
 *  Created on: 19 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_INTERCONNECT_H_
#define SRC_INTERCONNECT_H_

#include <stdio.h>

#include <mpi.h>

//enum BLOCK_TYPE { CPU, DEVICE0, DEVICE1, DEVICE2 }

class Interconnect {
public:
	Interconnect(int _sourceLocationNode, int _destinationLocationNode,
			int _sourceType, int _destinationType,
			int _lengthBorder,
			double* _sourceBlockBorder, double* _destinationExternalBorder);
	virtual ~Interconnect();

	void sendRecv(int locationNode);

private:
	int sourceLocationNode;
	int destinationLocationNode;

	int sourceType;
	int destinationType;

	int lengthBorder;

	double* sourceBlockBorder;
	double* destinationExternalBorder;

	MPI_Status status;
};

#endif /* SRC_INTERCONNECT_H_ */
