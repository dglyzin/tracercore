/*
 * Interconnect.cpp
 *
 *  Created on: 19 янв. 2015 г.
 *      Author: frolov
 */

#include "interconnect.h"

using namespace std;

Interconnect::Interconnect(int _sourceLocationNode, int _destinationLocationNode, int _borderLength,
		double* _sourceBlockBorder, double* _destinationExternalBorder, MPI_Comm* _pworkerComm ) {
	sourceLocationNode = _sourceLocationNode;
	destinationLocationNode = _destinationLocationNode;

	borderLength = _borderLength;

	sourceBlockBorder = _sourceBlockBorder;
	destinationExternalBorder = _destinationExternalBorder;

	request = new MPI_Request();

	status = new MPI_Status();

	flag = false;

	mpWorkerComm = _pworkerComm;
}

Interconnect::~Interconnect() {
	if( request != NULL )
		delete request;

	if( status != NULL )
		delete status;
}

void Interconnect::sendRecv(int locationNode) {
	/*
	 * Пересылка внутри блока.
	 */
	if(locationNode == sourceLocationNode && locationNode == destinationLocationNode)
		return;

	/*
	 * Если эту пеерсылку вызвал поток, который содержить информацию для пересылки.
	 * Фактически этот поток РЕАЛЬНО содержить блок имеющий исходную информацию. Блок-источник.
	 */
	if(locationNode == sourceLocationNode) {
		MPI_Isend(sourceBlockBorder, borderLength, MPI_DOUBLE, destinationLocationNode, 999, *mpWorkerComm, request);
		flag = true;
		return;
	}

	/*
	 * Есои пересылку вызвол поток, который должен принимать инфомацию от другого потока.
	 * Это поток РЕАЛЬНО имеет блок, которому необходима информация от другого блока.
	 */
	if(locationNode == destinationLocationNode) {
		MPI_Irecv(destinationExternalBorder, borderLength, MPI_DOUBLE, sourceLocationNode, 999, *mpWorkerComm, request);
		flag = true;
		return;
	}

	/*
	 * Если же это соединения вызовет поток, которые фактически не имет к нему отношения, то ничего не произойдет.
	 */
}

void Interconnect::wait() {
	if( flag ) {
		MPI_Wait(request, status);
		flag = false;
	}
}

void Interconnect::print() {
	/*cout << endl;
	cout << "Interconnect" << endl;
	cout << "	Source node               : " << sourceLocationNode << endl;
	cout << "	Destination node          : " << destinationLocationNode << endl;
	cout << "	Source memory address     : " << sourceBlockBorder << endl;
	cout << "	Destination memory address: " << destinationExternalBorder << endl;
	cout << endl;*/

	printf("\nInterconnect\n"
			"	Source node               : %d\n"
			"	Destination node          : %d\n"
			"	Source memory address     : %p\n"
			"	Destination memory address: %p\n"
			"\n", sourceLocationNode, destinationLocationNode, sourceBlockBorder, destinationExternalBorder);
}
