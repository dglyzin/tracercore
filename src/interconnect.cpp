/*
 * Interconnect.cpp
 *
 *  Created on: 19 янв. 2015 г.
 *      Author: frolov
 */

#include "interconnect.h"

using namespace std;

Interconnect::Interconnect(int _sourceLocationNode, int _destinationLocationNode,
		int _sourceType, int _destinationType,
		int _borderLength,
		double* _sourceBlockBorder, double* _destinationExternalBorder) {
	sourceLocationNode = _sourceLocationNode;
	destinationLocationNode = _destinationLocationNode;

	sourceType = _sourceType;
	destinationType = _destinationType;

	borderLength = _borderLength;

	sourceBlockBorder = _sourceBlockBorder;
	destinationExternalBorder = _destinationExternalBorder;

	if( sourceLocationNode == destinationLocationNode )
		request = NULL;
	else
		request = new MPI_Request();

	status = new MPI_Status();
}

Interconnect::~Interconnect() {
	if( request != NULL )
		delete request;
}

void Interconnect::sendRecv(int locationNode) {
	/*
	 * Пересылка внутри блока.
	 * На данный момент реализован не оптимальный способ передачи данных внутри потока.
	 * Данный передаются с помощью MPI-пересылок.
	 * В дальнейшем должена быть реализована "склейка" границ.
	 */
	if(locationNode == sourceLocationNode && locationNode == destinationLocationNode)
		return;

	/*
	 * Если эту пеерсылку вызвал поток, который содержить информацию для пересылки.
	 * Фактически этот поток РЕАЛЬНО содержить блок имеющий исходную информацию. Блок-источник.
	 */
	if(locationNode == sourceLocationNode) {
		MPI_Isend(sourceBlockBorder, borderLength, MPI_DOUBLE, destinationLocationNode, 999, MPI_COMM_WORLD, request);
		return;
	}

	/*
	 * Есои пересылку вызвол поток, который должен принимать инфомацию от другого потока.
	 * Это поток РЕАЛЬНО имеет блок, которому необходима информация от другого блока.
	 */
	if(locationNode == destinationLocationNode) {
		MPI_Irecv(destinationExternalBorder, borderLength, MPI_DOUBLE, sourceLocationNode, 999, MPI_COMM_WORLD, request);
		return;
	}

	/*
	 * Если же это соединения вызовет поток, которые фактически не имет к нему отношения, то ничего не произойдет.
	 */
}

void Interconnect::print() {
	cout << endl;
	cout << "Interconnect" << endl;
	cout << "	Source node               : " << sourceLocationNode << endl;
	cout << "	Destination node          : " << destinationLocationNode << endl;
	cout << "	Source memory address     : " << sourceBlockBorder << endl;
	cout << "	Destination memory address: " << destinationExternalBorder << endl;
	cout << endl;
}
