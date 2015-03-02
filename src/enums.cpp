/*
 * Enums.cpp
 *
 *  Created on: 02 февр. 2015 г.
 *      Author: frolov
 */

#include "enums.h"

int oppositeBorder(int side) {
	switch (side) {
		case TOP:
			return BOTTOM;
		case LEFT:
			return RIGHT;
		case BOTTOM:
			return TOP;
		case RIGHT:
			return LEFT;
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
