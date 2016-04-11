/*
 * gpu2d.h
 *
 *  Created on: 11 апр. 2016 г.
 *      Author: frolov
 */

#ifndef SRC_PROCESSINGUNIT_GPU_GPU2D_H_
#define SRC_PROCESSINGUNIT_GPU_GPU2D_H_

#include "gpu.h"

class GPU_2d: public GPU {
public:
	GPU_2d(int _deviceNumber);
	virtual ~GPU_2d();

	void computeBorder(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result, double** source,
			double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize);
	void computeCenter(func_ptr_t* mUserFuncs, unsigned short int* mCompFuncNumber, double* result, double** source,
			double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize);

	void printArray(double* array, int zCount, int yCount, int xCount, int cellSize);
};

#endif /* SRC_PROCESSINGUNIT_GPU_GPU2D_H_ */
