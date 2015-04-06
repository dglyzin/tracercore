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

bool isCPU(int type) {
	return type == CPU;
}

bool isGPU(int type) {
	return type == GPU;
}
