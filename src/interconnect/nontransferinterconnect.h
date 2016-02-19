/*
 * nontransferinterconnect.h
 *
 *  Created on: 19 февр. 2016 г.
 *      Author: frolov
 */

#ifndef SRC_INTERCONNECT_NONTRANSFERINTERCONNECT_H_
#define SRC_INTERCONNECT_NONTRANSFERINTERCONNECT_H_

#include "interconnect.h"

class NonTransferInterconnect: public Interconnect {
public:
	NonTransferInterconnect(int _sourceLocationNode, int _destinationLocationNode);
	virtual ~NonTransferInterconnect();

	void transfer();
	void wait();
};

#endif /* SRC_INTERCONNECT_NONTRANSFERINTERCONNECT_H_ */
