/*
 * gpu.cpp
 *
 *  Created on: 25 марта 2016 г.
 *      Author: frolov
 */

#include "gpu.h"

Gpu::Gpu(int _deviceNumber) :
		ProcessingUnit(_deviceNumber) {
}

Gpu::~Gpu() {
	deleteAllArrays();
}

