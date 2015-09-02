/*
 * dbconnector.h
 *
 *  Created on: Sep 2, 2015
 *      Author: dglyzin
 */

#ifndef DBCONNECTOR_H_
#define DBCONNECTOR_H_

void dbConnSetJobState(int mJobID, int state);
void dbConnSetJobPercentage(int mJobID, int state);
void dbConnStoreFileName(int jobId, char* fname);


#endif /* DBCONNECTOR_H_ */
