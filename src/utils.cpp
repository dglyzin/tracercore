/*
 * utils.cpp
 *
 *  Created on: 21 февр. 2016 г.
 *      Author: frolov
 */

#include "utils.h"

Utils::Utils() {
	// TODO Auto-generated constructor stub

}

Utils::~Utils() {
	// TODO Auto-generated destructor stub
}

int Utils::lastChar(const char* source, char ch) {
	int i = 0;
	int index = 0;
	while (source[i] != 0) {
		if (source[i] == ch)
			index = i;
		i++;
	}

	return index;
}

void Utils::cutToLastChar(char* result, const char* source, char ch) {
	int length = Utils::lastChar(source, ch);

	strncpy(result, source, length);

	result[length] = 0;
}
