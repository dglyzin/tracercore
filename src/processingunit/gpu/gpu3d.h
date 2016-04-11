/*
 * gpu3d.h
 *
 *  Created on: 11 апр. 2016 г.
 *      Author: frolov
 */

#ifndef SRC_PROCESSINGUNIT_GPU_GPU3D_H_
#define SRC_PROCESSINGUNIT_GPU_GPU3D_H_

#include "gpu.h"

class GPU_3d: public GPU {
public:
	GPU_3d(int _deviceNumber);
	virtual ~GPU_3d();
};

#endif /* SRC_PROCESSINGUNIT_GPU_GPU3D_H_ */
