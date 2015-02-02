/*
 * Enums.cpp
 *
 *  Created on: 02 февр. 2015 г.
 *      Author: frolov
 */

#include "Enums.h"

int getDeviceNumber(int blockType) {
	switch (blockType) {
		case DEVICE0:
			return 0;
		case DEVICE1:
			return 1;
		case DEVICE2:
			return 2;
		default:
			return 0;
	}
}

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
