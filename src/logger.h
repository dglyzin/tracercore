/*
 * logger.h
 *
 *  Created on: Mar 1, 2017
 *      Author: dglyzin
 */

#ifndef SRC_LOGGER_H_
#define SRC_LOGGER_H_

#include <string>
#include <ctime>
#include <sstream>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
using namespace std;

/*logging with timestamp*/
#define LL_INFO 0
#define LL_DEBUG 1

#define LOGLEVEL LL_DEBUG


string ToString(long);
string ToString(double);
string ToString(int);
string ToString(char*);
string ToString(unsigned long long);

void printwts(std::string message, time_t timestamp, int loglevel);
void printwcts(std::string message, int loglevel);

#endif /* SRC_LOGGER_H_ */
