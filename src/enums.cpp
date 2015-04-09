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
			return "LEFT";
		case RIGHT:
			return "RIGHT";
		case FRONT:
			return "FRONT";
		case BACK:
			return "BACK";
		case TOP:
			return "TOP";
		case BOTTOM:
			return "BOTTOM";
		default:
			return "ERROR SIDE";
	}
}

bool isCPU(int type) {
	return type == CPU;
}

bool isGPU(int type) {
	return type == GPU;
}
