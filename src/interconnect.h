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

#include "enums.h"

/*
 * Класс, отвечающий за пересылку данных между блоками.
 */

class Interconnect {
public:
	Interconnect(int _sourceLocationNode, int _destinationLocationNode, int _borderLength,
			double* _sourceBlockBorder, double* _destinationExternalBorder, MPI_Comm* _pworkerComm);
	virtual ~Interconnect();

	/*
	 * Переслать данные.
	 * Номера блоков неизвестны.
	 * Пересылаем по данным в конструкторе указателям и по данным о номере потока MPI.
	 * Передает информацию о потоке, который вызвал - его номер.
	 */
	void sendRecv(int locationNode);

	void print();

	void wait();

private:
	MPI_Comm* mpWorkerComm;
	/*
	 * Номер потока с исходными данными
	 */
	int sourceLocationNode;

	/*
	 * Номер потока, которому необходимо прислать данные
	 */
	int destinationLocationNode;

	/*
	 * Длина пересылаемого блока.
	 * Длина границы между блоками.
	 */
	int borderLength;

	/*
	 * Указатель на массив с исходными данными.
	 */
	double* sourceBlockBorder;

	/*
	 * Указатель на массив, куда нужно положить данные
	 */
	double* destinationExternalBorder;

	/*
	 * Служебные переменные
	 */
	MPI_Status* status;
	MPI_Request* request;

	bool flag;
};

#endif /* SRC_INTERCONNECT_H_ */
