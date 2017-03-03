/*
 * utils.cpp
 *
 *  Created on: 21 февр. 2016 г.
 *      Author: frolov
 */

#include "utils.h"

Utils::Utils() {
}

Utils::~Utils() {
}

int Utils::lastChar(const char* source, char ch, int num) {
	/*int i = 0;
	 int index = 0;
	 while (source[i] != 0) {
	 if (source[i] == ch)
	 index = i;
	 i++;
	 }

	 return index;*/

	int number = 0;

	int length = strlen(source);

	for (int i = length; i >= 0; --i) {
		if (source[i] == ch)
			if (++number == num)
				return i;
	}

	return -1;
}

void Utils::copyToLastChar(char* result, const char* source, char ch, int num) {
	int length = Utils::lastChar(source, ch, num) + 1;

	strncpy(result, source, length);

	result[length] = 0;
}

void Utils::copyToLastCharNotInc(char* result, const char* source, char ch, int num) {
	int length = Utils::lastChar(source, ch, num);

	strncpy(result, source, length);

	result[length] = 0;
}


void Utils::copyFromLastToEnd(char* result, const char* source, char ch, int num) {
	int pos = Utils::lastChar(source, ch, num) + 1;
	int length = strlen(source);

	strncpy(result, source + pos, length - pos);
}

void Utils::getFilePathForDraw(char* inputFile, char* saveFile, double currentTime, int plotVals) {
	//copyToLastChar(saveFile, inputFile, '/');
	//sprintf(saveFile, "%s%s%.8f%s", saveFile, FILE_NAME, currentTime, FILE_EXPANSION_DRAW);          
	copyToLastCharNotInc(saveFile, inputFile, '.');
	sprintf(saveFile, "%s-%.8f-%d%s", saveFile, currentTime, plotVals, FILE_EXPANSION_DRAW);
}

void Utils::getFilePathForLoad(char* inputFile, char* saveFile, double currentTime) {
	//copyToLastChar(saveFile, inputFile, '/');
	//sprintf(saveFile, "%s%s%.8f%s", saveFile, FILE_NAME, currentTime, FILE_EXPANSION_LOAD);
	copyToLastCharNotInc(saveFile, inputFile, '.');
	sprintf(saveFile, "%s-%.8f%s", saveFile, currentTime, FILE_EXPANSION_LOAD);
}

void Utils::getTracerFolder(char* binary, char* tracerFolder) {
	char temp[250];
	copyToLastCharNotInc(tracerFolder, binary, '/');
	copyToLastCharNotInc(temp, tracerFolder, '/');
	copyToLastCharNotInc(tracerFolder, temp, '/');
}

void Utils::getProjectFolder(char* projectFile, char* projectFolder) {
	copyToLastCharNotInc(projectFolder, projectFile, '/');
}


int Utils::oppositeBorder(int side) {
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

int Utils::getSide(int number) {
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

char* Utils::getSideName(int side) {
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

bool Utils::isCPU(int type) {
	return type == CPUNIT;
}

bool Utils::isGPU(int type) {
	return type == GPUNIT;
}
