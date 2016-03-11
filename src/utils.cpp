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

void Utils::copyFromLastToEnd(char* result, const char* source, char ch, int num) {
	int pos = Utils::lastChar(source, ch, num) + 1;
	int length = strlen(source);

	strncpy(result, source + pos, length - pos);
}