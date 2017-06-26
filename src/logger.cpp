/*
 * logger.cpp
 *
 *  Created on: Mar 1, 2017
 *      Author: dglyzin
 */
#include "logger.h"

using namespace std;

string ToString(long val)
{
    stringstream stream;
    stream << val;
    return stream.str();
}

string ToString(double val)
{
    stringstream stream;
    stream << val;
    return stream.str();
}

string ToString(char* val)
{
    stringstream stream;
    stream << val;
    return stream.str();
}


string ToString(int val)
{
    stringstream stream;
    stream << val;
    return stream.str();
}

string ToString(unsigned long long val)
{
    stringstream stream;
    stream << val;
    return stream.str();
}


//print with specified timestamp
void printwts(std::string message, time_t timestamp, int loglevel){
    //char* dt = ctime(&timestamp);
	if (loglevel>LOGLEVEL)
		return;

    tm *ltm = localtime(&timestamp);

    /*ofstream logFile;
    logFile.open("log", ios::app);*/

    // print various components of tm structure.
    printf("%02d-%02d %02d:%02d:%02d ", 1 + ltm->tm_mon, ltm->tm_mday, ltm->tm_hour, ltm->tm_min, ltm->tm_sec );

    std::cout << message;
    //logFile << message << endl;
    //logFile.close();
}

//print with current timestamp
void printwcts(std::string message, int loglevel){
    printwts(message,time(0), loglevel);
}




