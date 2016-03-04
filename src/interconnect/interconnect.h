/*
 * Interconnect.h
 *
 *  Created on: 19 янв. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_INTERCONNECT_INTERCONNECT_H_
#define SRC_INTERCONNECT_INTERCONNECT_H_

#include <stdio.h>

#include "../enums.h"

/*
 * Класс, отвечающий за пересылку данных между блоками.
 */
class Interconnect {
public:
	Interconnect(int _sourceLocationNode, int _destinationLocationNode);
	virtual ~Interconnect();

	virtual void transfer() = 0;
	virtual void wait() = 0;

	void print();

protected:
	int sourceLocationNode;
	int destinationLocationNode;

	void printNodeLocationInformation();

	virtual void printTypeInformation() = 0;
	virtual void printMemoryAddresInformation() = 0;
};

#endif /* SRC_INTERCONNECT_INTERCONNECT_H_ */
