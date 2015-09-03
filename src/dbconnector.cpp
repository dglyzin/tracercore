/*
 * dbconnector.cpp
 *
 *  Created on: Sep 2, 2015
 *      Author: dglyzin
 */
#include "enums.h"
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
	char stmtstring[256];
	if (state == JS_FINISHED)
	    sprintf(stmtstring, "UPDATE jobs SET state=%d, finishtime=NOW() WHERE id=%d", state, jobId);
	else
	    sprintf(stmtstring, "UPDATE jobs SET state=%d WHERE id=%d", state, jobId);

	execSQL(stmtstring);
}

void dbConnSetJobPercentage(int jobId, int percentage){
	char stmtstring[256];
	sprintf(stmtstring, "UPDATE jobs SET percentage=%d WHERE id=%d", percentage, jobId);
	execSQL(stmtstring);
}


void dbConnStoreFileName(int jobId, char* fname){
	try {
		  sql::Driver *driver;
		  sql::Connection *con;
		  sql::Statement *stmt;
		  sql::ResultSet *res;
		  /* Create a connection */
		  driver = get_driver_instance();
		  con = driver->connect("192.168.10.100", "cherry", "sho0ro0p");
		  /* Connect to the MySQL test database */
		  con->setSchema("cluster");
		  //find total number of files
		  int total = 0;
		  stmt = con->createStatement();
		  char stmtstring[512];
		  sprintf(stmtstring, "SELECT COUNT(job) AS NumberOfFiles FROM results WHERE job=%d", jobId);
		  res = stmt->executeQuery(stmtstring);
          if (res->next())
		      total = res->getInt(1);
		  delete stmt;
		  delete res;
		  //insert row with filename
          sprintf(stmtstring, "INSERT INTO results (job, num, fname) VALUES (%d, %d, '%s')", jobId, total, fname);
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


int dbConnGetUserStatus(int jobId){
	int result = 0;
	try {
		  sql::Driver *driver;
		  sql::Connection *con;
		  sql::Statement *stmt;
		  sql::ResultSet *res;
		  /* Create a connection */
		  driver = get_driver_instance();
		  con = driver->connect("192.168.10.100", "cherry", "sho0ro0p");
		  /* Connect to the MySQL test database */
		  con->setSchema("cluster");
		  //find total number of files

		  stmt = con->createStatement();
		  char stmtstring[512];
		  sprintf(stmtstring, "SELECT userstatus AS Ustat FROM jobs WHERE id=%d", jobId);
		  res = stmt->executeQuery(stmtstring);
          if (res->next())
		      result = res->getInt(1);
		  delete stmt;
		  delete res;
		  delete con;
		} catch (sql::SQLException &e) {
		  cout << "# ERR: SQLException in " << __FILE__;
		  cout << "(" << __FUNCTION__ << ") on line "   << __LINE__ << endl;
		  cout << "# ERR: " << e.what();
		  cout << " (MySQL error code: " << e.getErrorCode();
		  cout << ", SQLState: " << e.getSQLState() << " )" << endl;
		}
	return result;

}
