/*
 * dbconnector.cpp
 *
 *  Created on: Sep 2, 2015
 *      Author: dglyzin
 */

#include <stdlib.h>
#include <iostream>
#include <cstdio>

#include "dbconnector.h"

#include "mysql_connection.h"

#include <cppconn/driver.h>
#include <cppconn/exception.h>
#include <cppconn/resultset.h>
#include <cppconn/statement.h>

using namespace std;

void execSQL(char *sqlString){
	try {
	  sql::Driver *driver;
	  sql::Connection *con;
	  sql::Statement *stmt;
	  /* Create a connection */
	  driver = get_driver_instance();
	  con = driver->connect("192.168.10.100", "cherry", "sho0ro0p");
	  /* Connect to the MySQL test database */
	  con->setSchema("cluster");
	  stmt = con->createStatement();
	  stmt->execute(sqlString);
	  delete stmt;
	  delete con;
	} catch (sql::SQLException &e) {
	  cout << "# ERR: SQLException in " << __FILE__;
	  cout << "(" << __FUNCTION__ << ") on line "   << __LINE__ << endl;
	  cout << "# ERR: " << e.what();
	  cout << " (MySQL error code: " << e.getErrorCode();
	  cout << ", SQLState: " << e.getSQLState() << " )" << endl;
	}
}




void dbConnSetJobState(int jobId, int state){
	try {
	  sql::Driver *driver;
	  sql::Connection *con;
	  sql::Statement *stmt;

	  /* Create a connection */
	  driver = get_driver_instance();
	  con = driver->connect("192.168.10.100", "cherry", "sho0ro0p");
	  /* Connect to the MySQL test database */
	  con->setSchema("cluster");
	  char stmtstring[256];
	  //change state to computing
	  sprintf(stmtstring, "UPDATE jobs SET state=%d WHERE id=%d", state, jobId);
	  stmt = con->createStatement();
	  stmt->execute(stmtstring);
	  delete stmt;
	  delete con;
	} catch (sql::SQLException &e) {
	  cout << "# ERR: SQLException in " << __FILE__;
	  cout << "(" << __FUNCTION__ << ") on line "   << __LINE__ << endl;
	  cout << "# ERR: " << e.what();
	  cout << " (MySQL error code: " << e.getErrorCode();
	  cout << ", SQLState: " << e.getSQLState() << " )" << endl;
	}


}

void dbConnSetJobPercentage(int jobId, int percentage){
	try {
	  sql::Driver *driver;
	  sql::Connection *con;
	  sql::Statement *stmt;
	  /* Create a connection */
	  driver = get_driver_instance();
	  con = driver->connect("192.168.10.100", "cherry", "sho0ro0p");
	  /* Connect to the MySQL test database */
	  con->setSchema("cluster");
	  char stmtstring[256];
	  sprintf(stmtstring, "UPDATE jobs SET percentage=%d WHERE id=%d", percentage, jobId);
	  stmt = con->createStatement();
	  stmt->execute(stmtstring);
	  delete stmt;
	  delete con;
	} catch (sql::SQLException &e) {
	  cout << "# ERR: SQLException in " << __FILE__;
	  cout << "(" << __FUNCTION__ << ") on line "   << __LINE__ << endl;
	  cout << "# ERR: " << e.what();
	  cout << " (MySQL error code: " << e.getErrorCode();
	  cout << ", SQLState: " << e.getSQLState() << " )" << endl;
	}

}



