/*
 * cpu.cpp
 *
 *  Created on: 18 сент. 2015 г.
 *      Author: frolov
 */

#include "cpu.h"

CPU::CPU() {
	// TODO Auto-generated constructor stub

}

CPU::~CPU() {
	// TODO Auto-generated destructor stub
}

double* CPU::getDoubleArray(int size) {
	return new double [size];
}

double** CPU::getDoublePointerArray(int size) {
	return new double* [size];
}

int* CPU::getIntArray(int size) {
	return new int [size];
}

int** CPU::getIntPointerArray(int size) {
	return new int* [size];
}

void CPU::deleteDeviceSpecificArray(double* toDelete) {
	delete toDelete;
}

void CPU::deleteDeviceSpecificArray(double** toDelete) {
	delete toDelete;
}

void CPU::deleteDeviceSpecificArray(int* toDelete) {
	delete toDelete;
}

void CPU::deleteDeviceSpecificArray(int** toDelete) {
	delete toDelete;
}
