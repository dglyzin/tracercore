/*
 * utils.h
 *
 *  Created on: 21 февр. 2016 г.
 *      Author: frolov
 */

#ifndef SRC_UTILS_H_
#define SRC_UTILS_H_

#include <string.h>

class Utils {
public:
	Utils();
	virtual ~Utils();

	static int lastChar(const char* source, char ch, int num = 1);
	static void copyToLastChar(char* result, const char* source, char ch);
	static void cutToLastButOneChar(char* result, const char* source, char ch);
	static void cutFromLastToEnd(char* result, const char* source, char ch);
};

#endif /* SRC_UTILS_H_ */
