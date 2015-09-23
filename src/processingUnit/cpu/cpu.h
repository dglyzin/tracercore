/*
 * cpu.h
 *
 *  Created on: 18 сент. 2015 г.
 *      Author: frolov
 */

#ifndef SRC_PROCESSINGUNIT_CPU_CPU_H_
#define SRC_PROCESSINGUNIT_CPU_CPU_H_

class CPU: public ProcessingUnit {
public:
	CPU();
	virtual ~CPU();

	virtual void computeBorder(double* result, double** source, double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize) = 0;
	virtual void computeCenter(double* result, double** source, double time, double* parametrs, double** externalBorder, int zCount, int yCount, int xCount, int haloSize) = 0;

	void prepareBorder(double* result, double* source, int zStart, int zStop, int yStart, int yStop, int xStart, int xStop, int yCount, int xCount, int cellSize);


	double* getDoubleArray(int size);
	double** getDoublePointerArray(int size);

	int* getIntArray(int size);
	int** getIntPointerArray(int size);


	void deleteDeviceSpecificArray(double* toDelete);
	void deleteDeviceSpecificArray(double** toDelete);

	void deleteDeviceSpecificArray(int* toDelete);
	void deleteDeviceSpecificArray(int** toDelete);
};

#endif /* SRC_PROCESSINGUNIT_CPU_CPU_H_ */
