/*
 * utils.h
 *
 *  Created on: 21 февр. 2016 г.
 *      Author: frolov
 */

#ifndef SRC_UTILS_H_
#define SRC_UTILS_H_

#include <string.h>
#include <stdio.h>

#include "enums.h"

#define FILE_NAME_FOR_DRAW "project-"
#define FILE_NAME_FOR_LOAD "draw-"
#define FILE_EXPANSION ".bin"

class Utils {
public:
	Utils();
	virtual ~Utils();

	static int lastChar(const char* source, char ch, int num = 1);
	static void copyToLastChar(char* result, const char* source, char ch, int num = 1);
	static void copyFromLastToEnd(char* result, const char* source, char ch, int num = 1);

	static void getFilePathForDraw(char* inputFile, char* saveFile, double currentTime);
	static void getFilePathForLoad(char* inputFile, char* saveFile, double currentTime);

	static int oppositeBorder(int side);
	static int getSide(int number);
	static char* getSideName(int side);

	static bool isCPU(int type);
	static bool isGPU(int type);
};

#endif /* SRC_UTILS_H_ */
