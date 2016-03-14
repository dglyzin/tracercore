/*
 * Enums.cpp
 *
 *  Created on: 02 февр. 2015 г.
 *      Author: frolov
 */

#include "enums.h"

int oppositeBorder(int side) {
	switch (side) {
		case LEFT:
			return RIGHT;
		case RIGHT:
			return LEFT;
		case FRONT:
			return BACK;
		case BACK:
			return FRONT;
		case TOP:
			return BOTTOM;
		case BOTTOM:
			return TOP;
		default:
			return TOP;
	}
}

int getSide(int number) {
	switch (number) {
		case 0:
			return LEFT;
		case 1:
			return RIGHT;
		case 2:
			return FRONT;
		case 3:
			return BACK;
		case 4:
			return TOP;
		case 5:
			return BOTTOM;
		default:
			return LEFT;
	}
}

char* getSideName(int side) {
	switch (side) {
		case LEFT:
			return (char*) "LEFT";
		case RIGHT:
			return (char*) "RIGHT";
		case FRONT:
			return (char*) "FRONT";
		case BACK:
			return (char*) "BACK";
		case TOP:
			return (char*) "TOP";
		case BOTTOM:
			return (char*) "BOTTOM";
		default:
			return (char*) "ERROR SIDE";
	}
}

char* getMemoryTypeName(int type) {
	switch (type) {
		case NOT_ALLOC:
			return (char*) "NOT_ALLOC";
		case NEW:
			return (char*) "NEW";
		case CUDA_MALLOC:
			return (char*) "CUDA_MALLOC";
		case CUDA_MALLOC_HOST:
			return (char*) "CUDA_MALLOC_HOST";
		default:
			return (char*) "ERROR MEMORY TYPE";
	}
}

bool isCPU(int type) {
	return type == CPU_UNIT;
}

bool isGPU(int type) {
	return type == GPU_UNIT;
}
