/*
 * ismartcopy.h
 *
 *  Created on: 27 февр. 2017 г.
 *      Author: frolov
 */

#ifndef SRC_PROBLEM_ISMARTCOPY_H_
#define SRC_PROBLEM_ISMARTCOPY_H_

#include "../processingunit/processingunit.h"

class ISmartCopy {
public:
	ISmartCopy();
	virtual ~ISmartCopy();

	virtual void swapCopy(ProcessingUnit* pu, double** source, double** destination, unsigned long long size) = 0;
};

#endif /* SRC_PROBLEM_ISMARTCOPY_H_ */
