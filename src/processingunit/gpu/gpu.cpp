/*
 * gpu.cpp
 *
 *  Created on: 25 марта 2016 г.
 *      Author: frolov
 */

#include "gpu.h"

GPU::GPU(int _deviceNumber) :
		ProcessingUnit(_deviceNumber) {
}

GPU::~GPU() {
	deleteAllArrays();
}

