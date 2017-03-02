/*
 * Domain.cpp
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#include "domain.h"
#include <cassert>
#include <iostream>
#include <sstream>
#include <stdlib.h>


using namespace std;
#include <stdio.h>
#include <iostream>
#include <ctime>
#include <string>

/*logging with timestamp*/
#define LL_INFO 0
#define LL_DEBUG 1

#define LOGLEVEL LL_INFO

template <typename T>
string ToString(T val)
{
    stringstream stream;
    stream << val;
    return stream.str();
}

void printwts(std::string message, time_t timestamp, int loglevel){
    //char* dt = ctime(&timestamp);
	if (loglevel>LOGLEVEL)
		return;

    tm *ltm = localtime(&timestamp);

    // print various components of tm structure.
    printf("%02d-%02d %02d:%02d:%02d ", 1 + ltm->tm_mon, ltm->tm_mday, ltm->tm_hour, ltm->tm_min, ltm->tm_sec );


    std::cout << message;


}

void printwcts(std::string message, int loglevel){
    printwts(message,time(0), loglevel);
}

/*--------------------*/

Domain::Domain(int _world_rank, int _world_size, char* inputFile, char* binaryFileName) {
	//Get worker communicator and determine if there is python master
	mGlobalRank = _world_rank;

	MPI_Comm_split(MPI_COMM_WORLD, 1, _world_rank, &mWorkerComm);

	MPI_Comm_size(mWorkerComm, &mWorkerCommSize);
	MPI_Comm_rank(mWorkerComm, &mWorkerRank);

	if (mWorkerCommSize == _world_size)
		mPythonMaster = 0;
	else if (mWorkerCommSize == _world_size - 1)
		mPythonMaster = 1;
	else {
		mPythonMaster = 0;
		printf("Communicator size error!");
		assert(0);
	}

	//mJobId = _jobId;
    Utils::getTracerFolder(binaryFileName,	mTracerFolder);
    Utils::getProjectFolder(inputFile,	mProjectFolder);
	mJobState = JS_RUNNING;
	currentTime = 0;
	mStepCount = 0;

	mTimeStep = 0;
	stopTime = 0;

	mRepeatCount = 0;

	counterSaveTime = 0;

	dimension = 0;

	cpu = NULL;
	gpu = NULL;

	mGpuCount = GPU_COUNT;

	printf("\nPROBLEM TYPE ALWAYS = ORDINARY!!!\n");
	mProblenType = ORDINARY;

	readFromFile(inputFile);

	mAcceptedStepCount = 0;
	mRejectedStepCount = 0;
    

}

Domain::~Domain() {
	for (int i = 0; i < mBlockCount; ++i)
		delete mBlocks[i];
	delete mBlocks;

	for (int i = 0; i < mConnectionCount; ++i)
		delete mInterconnects[i];
	delete mInterconnects;

	MPI_Comm_free(&mWorkerComm);

	if (cpu)
		delete cpu;

	if (gpu) {
		for (int i = 0; i < mGpuCount; ++i) {
			delete gpu[i];
		}

		delete gpu;
	}

}

void Domain::compute(char* inputFile) {
    time_t now = time(0);
    double wnow = MPI_Wtime();
    double mnow = omp_get_wtime();

    printwts("Initital timestamp is " + ToString(now) +"\n" , now, LL_INFO );

    printwcts("Tracer root folder: " + ToString(mTracerFolder)+ "\n", LL_INFO);
    printwcts("Project folder: " + ToString(mProjectFolder)+ "\n", LL_INFO);
    printwcts("Computing from " + ToString(currentTime) + " to " + ToString(stopTime) +
    		       " with step "+ ToString(mTimeStep)+"\n", LL_INFO);
    printwcts("Computation started, worker #"+ ToString(mWorkerRank) +"\n", LL_INFO);
    printwcts("solver stage count: " + ToString(mSolverInfo->getStageCount())+ "\n", LL_INFO);




	if (mSolverInfo->isFSAL())
		initSolvers();

//	Порядок работы
//	                1. WORLD+COMP                          2. WORLD ONLY
//+	1. WORLD Bcast user-status, источник - world-0    |    +
//+	             xx.  идет расчет шага, используется только COMP
//пока нет	2. WORLD Allreduce compute-status                 |    +
//пока нет       xx.  идет расчет ошибки, используется только COMP
//+	5. accept/reject, comp-0 -> world-0               |    -
//+	6. new timestep, comp-0 -> world-0                |    -
//+	7. ready to collect data, comp-0 -> world-0       |    -
//+	8. WORLD collect data                             |    +
//  9. stop/continue comp-0 -> world-0                |    -

	double computeInterval = stopTime - currentTime;
	int percentage = 0;

	//1.
	mUserStatus = US_RUN;
	mJobState = JS_RUNNING;
	MPI_Bcast(&mUserStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (mPythonMaster && (mWorkerRank == 0))
		MPI_Send(&mJobState, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

	printwcts( "Initial user status received: " + ToString(userStatus) + "\n", LL_INFO );

	// TODO если пользователь остановил расчеты, то необходимо выполнить сохранение для загузки состояния (saveStateForLoad)
	while ((mUserStatus != US_STOP) && (mJobState == JS_RUNNING)) {
		nextStep();

		if (mPythonMaster && (mWorkerRank == 0)) {
			MPI_Send(&mLastStepAccepted, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
			MPI_Send(&mTimeStep, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			MPI_Send(&currentTime, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		}

		//printBlocksToConsole();

		//here is the logic of collecting raw data to one node and saving it to disk
		//worker 0 will do it if there is no python master
		//if python master is present, it receives all the data for all blocks,
		//creates and saves pictures, saves raw data and stores filenames to db

		int newPercentage = 100.0 * (1.0 - (stopTime - currentTime) / computeInterval);
		int percentChanged = newPercentage > percentage;
		if (mPythonMaster && (mWorkerRank == 0))
			MPI_Send(&percentChanged, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

		if (percentChanged) {
			time_t now2 = time(0);
		    double wnow2 = MPI_Wtime();
		    double mnow2 = omp_get_wtime();

			percentage = newPercentage;
			if (mPythonMaster && (mWorkerRank == 0)){
				MPI_Send(&percentage, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
	            double percenttime = wnow2-wnow;
			    MPI_Send(&percenttime, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
			    printwcts("Done " + ToString(percentage) + "% in " + ToString((mnow2-mnow))+ " seconds, and wtime gives " + ToString((wnow2-wnow))+ " seconds, ETA = "+ ToString((100-percentage)*(mnow2-mnow)) + " seconds\n" , LL_INFO);

			}
            if (!mPythonMaster && (mWorkerRank == 0)){
            	//printwcts("Done " + ToString(percentage) + "% in " + ToString((int) (now2-now))+ " seconds, and wtime gives " + ToString((wnow2-wnow))+ " seconds, and omp_wtime gives " + ToString((mnow2-mnow))+ " seconds, ratio is "  + ToString((wnow2-wnow)/(mnow2-mnow)) + " \n", LL_INFO);
            	printwcts("Done " + ToString(percentage) + "% in " + ToString((mnow2-mnow))+ " seconds, and wtime gives " + ToString((wnow2-wnow))+ " seconds, ETA = "+ ToString((100-percentage)*(mnow2-mnow)) + " seconds\n" , LL_INFO);
            	//system("ls -la");
            }
            now = now2;
            wnow = wnow2;
            mnow = mnow2;
		}

		counterSaveTime += mTimeStep;

		int readyToSave = (saveInterval != 0) && (counterSaveTime > saveInterval);
		if (mPythonMaster && (mWorkerRank == 0))
			MPI_Send(&readyToSave, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		if (!(currentTime < stopTime))
			mJobState = JS_FINISHED;
		if (mPythonMaster && (mWorkerRank == 0))
			MPI_Send(&mJobState, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

		if (readyToSave) {
			counterSaveTime = 0;
			saveStateForDraw(inputFile);
		}

		//check for termination request
		if (mPythonMaster && (mWorkerRank == 0)) {
			MPI_Bcast(&mUserStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
		}

	}
	printwcts("Computation finished for worker #" + ToString(mWorkerRank) + "\n", LL_INFO);

	//if ((mWorkerRank == 0)&&(!mPythonMaster))
	//    setDbJobState(JS_FINISHED);


	char comline [250];
	sprintf(comline, "python %s/hybriddomain/fakejobrunner.py", mTracerFolder );
	printwcts("comm line = "+ToString(comline) + "\n",LL_INFO);
	system(comline);


	char comline [250];
	sprintf(comline, "python %s/hybriddomain/fakejobrunner.py", mTracerFolder );
	printwcts("comm line = "+ToString(comline) + "\n",LL_INFO);
	system(comline);

}

void Domain::initSolvers() {
	computeStage(SOLVER_INIT_STAGE);
}

void Domain::computeStage(int stage) {
	prepareData(stage);

	for (int i = 0; i < mConnectionCount; ++i)
		mInterconnects[i]->transfer();

	computeOneStepCenter(stage);

	for (int i = 0; i < mConnectionCount; ++i)
		mInterconnects[i]->wait();

	computeOneStepBorder(stage);

	prepareNextStageArgument(stage);

	/*printf("\nstage #%d\n", stage);
	printBlocksToConsole();*/
}

void Domain::nextStep() {
	//int totalGridElements = getGridElementCount();
	//последовательно выполняем все стадии метода
	for (int stage = 0; stage < mSolverInfo->getStageCount(); stage++)
		computeStage(stage);

	//!!! Собрать мастеру ошибки
	//!!! если ошибки нет, продолжать
	if (mSolverInfo->isVariableStep()) {
		// MPI inside!
		double error = collectError();

		printf("step error = %f\n", error);

		//bool isErrorPermissible = mSolverInfo->isErrorPermissible(error, totalGridElementCount);
		bool isErrorPer = isErrorPermissible(error);

		if (isErrorPer) {
			if(currentTime + mTimeStep > stopTime) {
				mUserStatus = US_STOP;
				return;
			}

			currentTime += mTimeStep;
		}

		//!!! только 0, рассылать
		//mTimeStep = mSolverInfo->getNewStep(mTimeStep, error, totalGridElementCount);
		mTimeStep = getNewStep(error);
		printf("new time step = %f\n", mTimeStep);

		//!!! только 0, рассылать
		if (isErrorPer) {
			confirmStep(); //uses new timestep
			mAcceptedStepCount++;
			currentTime += mTimeStep;
			//cout<<"Step accepted\n"<<endl;
		} else {
			rejectStep(); //uses new timestep
			mRejectedStepCount++;
			cout << "Step rejected!\n" << endl;
		}
	} else { //constant step
		confirmStep();
		mAcceptedStepCount++;
		currentTime += mTimeStep;

	}
}

void Domain::prepareDeviceData(int deviceType, int deviceNumber, int stage) {
	for (int i = 0; i < mBlockCount; ++i)
		/*mBlocks[i]->getBlockType() == deviceType && mBlocks[i]->getDeviceNumber() == deviceNumber*/
		if (mBlocks[i]->isBlockType(deviceType) && mBlocks[i]->isDeviceNumber(deviceNumber)) {
			//printf("\nSuccses\n");
			mBlocks[i]->prepareStageData(stage);
		}
}

void Domain::processDeviceBlocksBorder(int deviceType, int deviceNumber, int stage) {
	for (int i = 0; i < mBlockCount; ++i)
		if (mBlocks[i]->isBlockType(deviceType) && mBlocks[i]->isDeviceNumber(deviceNumber)) {
			//cout << endl << "ERROR! PROCESS DEVICE!" << endl;
			mBlocks[i]->computeStageBorder(stage, currentTime);
		}
}

void Domain::processDeviceBlocksCenter(int deviceType, int deviceNumber, int stage) {
	for (int i = 0; i < mBlockCount; ++i)
		if (mBlocks[i]->isBlockType(deviceType) && mBlocks[i]->isDeviceNumber(deviceNumber)) {
			//cout << endl << "ERROR! PROCESS DEVICE!" << endl;
			mBlocks[i]->computeStageCenter(stage, currentTime);
		}
}
void Domain::prepareDeviceArgument(int deviceType, int deviceNumber, int stage) {
	for (int i = 0; i < mBlockCount; ++i)
		if (mBlocks[i]->isBlockType(deviceType) && mBlocks[i]->isDeviceNumber(deviceNumber)) {
			//cout << endl << "ERROR! PROCESS DEVICE!" << endl;
			mBlocks[i]->prepareArgument(stage, mTimeStep);
		}
}

double Domain::getDeviceError(int deviceType, int deviceNumber) {
	double error = 0;
	for (int i = 0; i < mBlockCount; ++i)
		if (mBlocks[i]->isBlockType(deviceType) && mBlocks[i]->isDeviceNumber(deviceNumber)) {
			//cout << endl << "ERROR! PROCESS DEVICE!" << endl;
			error += mBlocks[i]->getStepError(mTimeStep);
		}
	return error;
}

void Domain::prepareData(int stage) {
	/*#pragma omp task
	 prepareDeviceData(GPU_UNIT, 0, stage);
	 #pragma omp task
	 prepareDeviceData(GPU_UNIT, 1, stage);
	 #pragma omp task
	 prepareDeviceData(GPU_UNIT, 2, stage);*/

	for (int i = 0; i < mGpuCount; ++i) {
#pragma omp task
		prepareDeviceData(GPUNIT, i, stage);
	}

	prepareDeviceData(CPUNIT, 0, stage);

#pragma omp taskwait
}

void Domain::computeOneStepBorder(int stage) {
	/*#pragma omp task
	 processDeviceBlocksBorder(GPU_UNIT, 0, stage);
	 #pragma omp task
	 processDeviceBlocksBorder(GPU_UNIT, 1, stage);
	 #pragma omp task
	 processDeviceBlocksBorder(GPU_UNIT, 2, stage);*/

	for (int i = 0; i < mGpuCount; ++i) {
#pragma omp task
		processDeviceBlocksBorder(GPUNIT, i, stage);
	}

	processDeviceBlocksBorder(CPUNIT, 0, stage);
}

void Domain::prepareNextStageArgument(int stage) {
	/*#pragma omp task
	 prepareDeviceArgument(GPU_UNIT, 0, stage);
	 #pragma omp task
	 prepareDeviceArgument(GPU_UNIT, 1, stage);
	 #pragma omp task
	 prepareDeviceArgument(GPU_UNIT, 2, stage);*/

	for (int i = 0; i < mGpuCount; ++i) {
#pragma omp task
		prepareDeviceArgument(GPUNIT, i, stage);
	}

	prepareDeviceArgument(CPUNIT, 0, stage);
}

void Domain::computeOneStepCenter(int stage) {
	/*#pragma omp task
	 processDeviceBlocksCenter(GPU_UNIT, 0, stage);
	 #pragma omp task
	 processDeviceBlocksCenter(GPU_UNIT, 1, stage);
	 #pragma omp task
	 processDeviceBlocksCenter(GPU_UNIT, 2, stage);*/

	for (int i = 0; i < mGpuCount; ++i) {
#pragma omp task
		processDeviceBlocksCenter(GPUNIT, i, stage);
	}

	processDeviceBlocksCenter(CPUNIT, 0, stage);
}

//TODO next two methods are not parallel!
void Domain::confirmStep() {
	mLastStepAccepted = 1;
	for (int i = 0; i < mBlockCount; ++i) {
		mBlocks[i]->confirmStep(mTimeStep);
	}
}

void Domain::rejectStep() {
	mLastStepAccepted = 0;
	for (int i = 0; i < mBlockCount; ++i) {
		mBlocks[i]->rejectStep(mTimeStep);
	}
}

double Domain::collectError() {
	/*double err1, err2, err3;
	 err1 = err2 = err3 = 0;
	 //1. Get total error for current node
	 //TODO сделать циклом. как в функциях расчета
	 #pragma omp task
	 err1 = getDeviceError(GPU_UNIT, 0);
	 #pragma omp task
	 err2 = getDeviceError(GPU_UNIT, 1);
	 #pragma omp task
	 err3 = getDeviceError(GPU_UNIT, 2);

	 double nodeError = getDeviceError(CPU_UNIT, 0);

	 #pragma omp taskwait
	 nodeError += err1 + err2 + err3;

	 //2. Collect errors from all nodes
	 double totalError = nodeError;

	 //TODO MPI_ALLREDUCE
	 return totalError;*/

	double cpuError = 0;
	double* gpuError = new double[mGpuCount];

	for (int i = 0; i < mGpuCount; ++i) {
#pragma omp task
		gpuError[i] = getDeviceError(GPUNIT, i);
	}

	cpuError = getDeviceError(CPUNIT, 0);
#pragma omp taskwait

	double nodeError = cpuError;
	for (int i = 0; i < mGpuCount; ++i) {
		nodeError += gpuError[i];
	}

	double absError = 0;

	MPI_Reduce(&nodeError, &absError, 1, MPI_DOUBLE, MPI_SUM, 0, mWorkerComm);

	delete gpuError;

	return absError;
}

bool Domain::isErrorPermissible(double error) {
	int isErrorPermissible = 0;

	if (mWorkerRank == 0) {
		isErrorPermissible = (int)(mSolverInfo->isErrorPermissible(error, totalGridElementCount));
	}

	MPI_Bcast(&isErrorPermissible, 1, MPI_INT, 0, mWorkerComm);
	return (bool)(isErrorPermissible);
}

double Domain::getNewStep(double error) {
	double newStep = 0;

	if (mWorkerRank == 0) {
		newStep = mSolverInfo->getNewStep(mTimeStep, error, totalGridElementCount);
	}

	MPI_Bcast(&newStep, 1, MPI_DOUBLE, 0, mWorkerComm);
	return mSolverInfo->getNewStep(mTimeStep, error, totalGridElementCount);
}

void Domain::printBlocksToConsole() {
	for (int i = 0; i < mWorkerCommSize; ++i) {
		if (mWorkerRank == i) {
			for (int j = 0; j < mBlockCount; ++j) {
				mBlocks[j]->print();
			}
		}

		MPI_Barrier(mWorkerComm);
	}
}

void Domain::readFromFile(char* path) {
	ifstream in;
	in.open(path, ios::binary);

	readFileStat(in);
	readTimeSetting(in);
	readSaveInterval(in);
	readGridSteps(in);

	in.read((char*) &dimension, SIZE_INT);
	createProcessigUnit();

	readCellAndHaloSize(in);
	readSolverIndex(in);
	readSolverTolerance(in);

	initSolverInfo();

	createBlock(in);

	createInterconnect(in);

	for (int i = 0; i < mBlockCount; ++i)
		mBlocks[i]->moveTempBorderVectorToBorderArray();

	totalGridNodeCount = getGridNodeCount();
	totalGridElementCount = getGridElementCount();

	blockAfterCreate();

	//printBlocksToConsole();
}

void Domain::readFileStat(ifstream& in) {
	char fileType;
	char versionMajor;
	char versionMinor;

	in.read((char*) &fileType, 1);
	in.read((char*) &versionMajor, 1);
	in.read((char*) &versionMinor, 1);

	/*cout << endl;
	 cout << "file type:     " << (unsigned int)fileType << endl;
	 cout << "version major: " << (unsigned int)versionMajor << endl;
	 cout << "version minor: " << (unsigned int)versionMinor << endl;*/
}

void Domain::readTimeSetting(ifstream& in) {
	in.read((char*) &startTime, SIZE_DOUBLE);
	in.read((char*) &stopTime, SIZE_DOUBLE);
	in.read((char*) &mTimeStep, SIZE_DOUBLE);

	/*cout << "start time:    " << startTime << endl;
	 cout << "stop time:     " << stopTime << endl;
	 cout << "step time:     " << timeStep << endl;*/
}

void Domain::readSaveInterval(ifstream& in) {
	in.read((char*) &saveInterval, SIZE_DOUBLE);

	//cout << "save interval: " << saveInterval << endl;
}

void Domain::readGridSteps(ifstream& in) {
	in.read((char*) &mDx, SIZE_DOUBLE);
	in.read((char*) &mDy, SIZE_DOUBLE);
	in.read((char*) &mDz, SIZE_DOUBLE);

	/*cout << "dx:            " << mDx << endl;
	 cout << "dy:            " << mDy << endl;
	 cout << "dz:            " << mDz << endl;*/
}

void Domain::readCellAndHaloSize(ifstream& in) {
	in.read((char*) &mCellSize, SIZE_INT);
	in.read((char*) &mHaloSize, SIZE_INT);

	/*cout << "cell size:     " << mCellSize << endl;
	 cout << "halo size:     " << mHaloSize << endl;*/
}

void Domain::readSolverIndex(std::ifstream& in) {
	in.read((char*) &mSolverIndex, SIZE_INT);
	//cout << "Solver index:  " << mSolverIndex << endl;
}

void Domain::readSolverTolerance(std::ifstream& in) {
	in.read((char*) &mAtol, SIZE_DOUBLE);
	//cout << "Solver absolute tolerance:  " << mAtol << endl;
	in.read((char*) &mRtol, SIZE_DOUBLE);
	//cout << "Solver relative tolerance:  " << mRtol << endl;
}

void Domain::readBlockCount(ifstream& in) {
	in.read((char*) &mBlockCount, SIZE_INT);

	//cout << "block count:   " << mBlockCount << endl;
}

void Domain::readConnectionCount(ifstream& in) {
	in.read((char*) &mConnectionCount, SIZE_INT);

	//cout << "connection count:   " << mConnectionCount << endl;
}

/*
 * Чтение одного конкретного блока.
 * Эта функция заносит в блок лишь базовую инфомармацию.
 *
 * Размеры
 * Координаты
 * Номер потока-создателя
 * Тип блока
 *
 * После чтения блок будет считать, что ни с кем не связан.
 * Не будет готовить информацию для пересылки и не будет считываеть ее из других источников.
 */
Block* Domain::readBlock(ifstream& in, int idx, int dimension) {
	Block* resBlock;

	int node;
	int deviceType;
	int deviceNumber;

	int* count = new int[3];
	count[0] = count[1] = count[2] = 1;

	int* offset = new int[3];
	offset[0] = offset[1] = offset[2] = 0;

	int total = 1;

	in.read((char*) &node, SIZE_INT);
	in.read((char*) &deviceType, SIZE_INT);
	in.read((char*) &deviceNumber, SIZE_INT);

	/*cout << endl;
	 cout << "Block #" << idx << endl;
	 cout << "	dimension:     " << dimension << endl;
	 cout << "	node:          " << node << endl;
	 cout << "	device type:   " << deviceType << endl;
	 cout << "	device number: " << deviceNumber << endl;*/

	for (int j = 0; j < dimension; ++j) {
		in.read((char*) &offset[j], SIZE_INT);
		//cout << "	offset" << j << ":           " << offset[j] << endl;
	}

	for (int j = 0; j < dimension; ++j) {
		in.read((char*) &count[j], SIZE_INT);
		//cout << "	count" << j << ":            " << count[j] << endl;
		total *= count[j];
	}

	unsigned short int* initFuncNumber = new unsigned short int[total];
	unsigned short int* compFuncNumber = new unsigned short int[total];

	in.read((char*) initFuncNumber, total * SIZE_UN_SH_INT);
	in.read((char*) compFuncNumber, total * SIZE_UN_SH_INT);

	//cout << "Init func number###:" << endl;
	/*for (int idxY = 0; idxY < count[1]; ++idxY) {
	 for (int idxX = 0; idxX < count[0]; ++idxX)
	 cout << initFuncNumber[idxY*count[0]+idxX] << " ";
	 cout << endl;
	 }
	 cout << endl;*/

	//cout << "Comp func number:" << endl;
	/*for (int idxY = 0; idxY < count[1]; ++idxY) {
	 for (int idxX = 0; idxX < count[0]; ++idxX)
	 cout << compFuncNumber[idxY*count[0]+idxX] << " ";
	 cout << endl;
	 }
	 cout << endl;*/

	if (node == mWorkerRank) {
		ProcessingUnit* pu = NULL;

		if (deviceType == 0)  //CPU BLOCK
			switch (deviceNumber) {
				case 0:
					pu = cpu;
					break;

				default:
					printf("Invalid block device number for CPU!\n");
					assert(false);
					break;
			}
		else if (deviceType == 1) {  //GPU BLOCK
			/*switch (deviceNumber) {
			 case 0:
			 pu = gpu0;
			 break;

			 case 1:
			 pu = gpu1;
			 break;

			 case 2:
			 pu = gpu2;
			 break;

			 default:
			 printf("Invalid block device number for GPU!\n");
			 assert(false);
			 break;
			 }*/
			if (deviceNumber >= mGpuCount) {
				printf("Invalid block device number for GPU!\n");
				assert(false);
			}

			pu = gpu[deviceNumber];
		} else {
			printf("Invalid block type!\n");
			assert(false);
		}

		resBlock = new RealBlock(node, dimension, count[0], count[1], count[2], offset[0], offset[1], offset[2],
				mCellSize, mHaloSize, idx, pu, initFuncNumber, compFuncNumber, mProblenType, mSolverIndex, mAtol,
				mRtol);
	} else {
		//resBlock =  new BlockNull(idx, dimension, count[0], count[1], count[2], offset[0], offset[1], offset[2], node, deviceNumber, mHaloSize, mCellSize);
		resBlock = new NullBlock(node, dimension, count[0], count[1], count[2], offset[0], offset[1], offset[2],
				mCellSize, mHaloSize);
	}

	delete initFuncNumber;
	delete compFuncNumber;

	//resBlock->createSolver(mSolverIndex, mAtol, mRtol);

	return resBlock;
}

/*
 * Чтение соединения.
 */
Interconnect* Domain::readConnection(ifstream& in) {
	int dimension;
	int sourceBlock;
	int destinationBlock;

	int sourceSide;
	int destinationSide;

	int* length = new int[2];
	length[0] = length[1] = 1;

	int* offsetSource = new int[2];
	offsetSource[0] = offsetSource[1] = 0;

	int* offsetDestination = new int[2];
	offsetDestination[0] = offsetDestination[1] = 0;

	in.read((char*) &dimension, SIZE_INT);
	//cout << endl;
	//cout << "Interconnect #<NONE>" << endl;

	for (int j = 2 - dimension; j < 2; ++j) {
		in.read((char*) &length[j], SIZE_INT);
		//cout << "	length" << j << ":           " << length[j] << endl;
	}

	in.read((char*) &sourceBlock, SIZE_INT);
	in.read((char*) &destinationBlock, SIZE_INT);
	in.read((char*) &sourceSide, SIZE_INT);
	in.read((char*) &destinationSide, SIZE_INT);

	/*cout << "	source block:      " << sourceBlock << endl;
	 cout << "	destination block: " << destinationBlock << endl;
	 cout << "	source side:       " << sourceSide << endl;
	 cout << "	destination side:  " << destinationSide << endl;*/

	for (int j = 2 - dimension; j < 2; ++j) {
		in.read((char*) &offsetSource[j], SIZE_INT);
		//cout << "	offsetSource" << j << ":            " << offsetSource[j] << endl;
	}

	for (int j = 2 - dimension; j < 2; ++j) {
		in.read((char*) &offsetDestination[j], SIZE_INT);
		//cout << "	offsetDestnation" << j << ":        " << offsetDestination[j] << endl;
	}

	double* sourceData = mBlocks[sourceBlock]->addNewBlockBorder(mBlocks[destinationBlock], Utils::getSide(sourceSide),
			offsetSource[0], offsetSource[1], length[0], length[1]);
	double* destinationData = mBlocks[destinationBlock]->addNewExternalBorder(mBlocks[sourceBlock],
			Utils::getSide(destinationSide), offsetDestination[0], offsetDestination[1], length[0], length[1],
			sourceData);

	int sourceNode = mBlocks[sourceBlock]->getNodeNumber();
	int destinationNode = mBlocks[destinationBlock]->getNodeNumber();

	int borderLength = length[0] * length[1] * mCellSize * mHaloSize;

	//cout << endl << "ERROR sorceData = destinationData = NULL!!!" << endl;

	delete length;
	delete offsetSource;
	delete offsetDestination;

	//return new Interconnect(sourceNode, destinationNode, borderLength, sourceData, destinationData, &mWorkerComm);
	return getInterconnect(sourceNode, destinationNode, borderLength, sourceData, destinationData);
}

/*
 * Получение общего количества узлов сетки.
 * Сумма со всех блоков.
 */
int Domain::getGridNodeCount() {
	int count = 0;
	for (int i = 0; i < mBlockCount; ++i)
		count += mBlocks[i]->getGridNodeCount();

	return count;
}

/*
 * Получение общего количества элементов сетки.
 * Сумма со всех блоков.
 */
int Domain::getGridElementCount() {
	int count = 0;
	for (int i = 0; i < mBlockCount; ++i)
		count += mBlocks[i]->getGridElementCount();

	return count;
}

/*
 * Заново вычисляется количество повторений для вычислений.
 * Функция носит исключетельно статистический смысл (на данный момент).
 */
int Domain::getRepeatCount() {
	return mRepeatCount;
}

/*
 * Количество блоков, имеющих тип "центальный процессор".
 */
int Domain::getCpuBlockCount() {
	int count = 0;
	for (int i = 0; i < mBlockCount; ++i)
		if (Utils::isCPU(mBlocks[i]->getBlockType()))
			count++;

	return count;
}

/*
 * Количество блоков, имеющих тип "видеокарта".
 */
int Domain::getGpuBlockCount() {
	int count = 0;
	for (int i = 0; i < mBlockCount; ++i)
		if (Utils::isGPU(mBlocks[i]->getBlockType()))
			count++;

	return count;
}

/*
 * Количество реальных блоков на этом потоке.
 */
int Domain::realBlockCount() {
	int count = 0;
	for (int i = 0; i < mBlockCount; ++i)
		if (mBlocks[i]->isRealBlock())
			count++;

	return count;
}

void Domain::saveStateForDraw(char* inputFile) {
	char* saveFile = new char[250];
	Utils::getFilePathForDraw(inputFile, saveFile, currentTime);

	saveGeneralInfo(saveFile);
	saveStateForDrawByBlocks(saveFile);

	delete saveFile;
}

void Domain::saveStateForLoad(char* inputFile) {
	char* saveFile = new char[250];
	Utils::getFilePathForLoad(inputFile, saveFile, currentTime);

	saveGeneralInfo(saveFile);
	saveStateForLoadByBlocks(saveFile);

	delete saveFile;
}

void Domain::saveStateForDrawDenseOutput(char* inputFile) {
	char* saveFile = new char[250];
	Utils::getFilePathForDraw(inputFile, saveFile, stopTime);

	saveGeneralInfo(saveFile);
	saveStateForDrawDenseOutputByBlocks(saveFile, stopTime);

	delete saveFile;
}

void Domain::saveGeneralInfo(char* path) {
	if (mGlobalRank == 0) {
		ofstream out;
		out.open(path, ios::binary);

		char save_file_code = SAVE_FILE_CODE;
		char version_major = VERSION_MAJOR;
		char version_minor = VERSION_MINOR;

		out.write((char*) &save_file_code, SIZE_CHAR);
		out.write((char*) &version_major, SIZE_CHAR);
		out.write((char*) &version_minor, SIZE_CHAR);

		out.write((char*) &currentTime, SIZE_DOUBLE);
		out.write((char*) &mTimeStep, SIZE_DOUBLE);

		out.close();
	}
}

void Domain::saveStateForDrawByBlocks(char* path) {
	for (int i = 0; i < mBlockCount; ++i) {
		mBlocks[i]->saveStateForDraw(path);
		MPI_Barrier(mWorkerComm);
	}
}

void Domain::saveStateForLoadByBlocks(char* path) {
	for (int i = 0; i < mBlockCount; ++i) {
		mBlocks[i]->saveStateForLoad(path);
		MPI_Barrier(mWorkerComm);
	}
}

void Domain::saveStateForDrawDenseOutputByBlocks(char* path, double requiredTime) {
	for (int i = 0; i < mBlockCount; ++i) {
		mBlocks[i]->saveStateForDrawDenseOutput(path, mTimeStep, getTethaForDenseOutput(requiredTime));
		MPI_Barrier(mWorkerComm);
	}
}

double Domain::getTethaForDenseOutput(double requiredTime) {
	return (requiredTime - currentTime) / mTimeStep;
}

void Domain::loadStateFromFile(char* dataFile) {
	ifstream in;
	in.open(dataFile, ios::binary);

	char save_file_code;
	char version_major;
	char version_minor;
	double fileCurrentTime;

	in.read((char*) &save_file_code, SIZE_CHAR);
	in.read((char*) &version_major, SIZE_CHAR);
	in.read((char*) &version_minor, SIZE_CHAR);

	in.read((char*) &fileCurrentTime, SIZE_DOUBLE);
	in.read((char*) &mTimeStep, SIZE_DOUBLE);

	currentTime = fileCurrentTime;

	for (int i = 0; i < mBlockCount; ++i) {
		mBlocks[i]->loadState(in);
	}

	in.close();
}

void Domain::setStopTime(double _stopTime) {
	stopTime = _stopTime;
}

void Domain::printStatisticsInfo(char* inputFile, char* outputFile, double calcTime, char* statisticsFile) {
	//cout << endl << "PRINT STATISTIC INFO DOESN'T WORK" << endl;

	if (mWorkerRank == 0) {
		int count = 0;
		for (int i = 0; i < mBlockCount; ++i) {
			count += mBlocks[i]->getGridElementCount();

			//mBlocks[i]->printGeneralInformation();
		}

		int stepCount = mRejectedStepCount + mAcceptedStepCount;
		double speed = (double) (count) * stepCount / calcTime / 1000000;

		printf("\n\nSteps accepted: %d\nSteps rejected: %d\n", mAcceptedStepCount, mRejectedStepCount);
		printf("Time: %.2f\nElement count: %d\nPerformance (10^6): %.2f\n\n", calcTime, count, speed);

		//ofstream out;
		//out.open("/home/frolov2/Tracer_project/stat", ios::app);

		/*FILE* out;
		 out = fopen("/home/frolov2/Tracer_project/statistic", "a");

		 double speed = (double)(count) * stepCount / calcTime / 1000000;
		 int side = (int)sqrt( ( (double)count ) / mCellSize );

		 fprintf(out, "%-12d %-8d %-2d    %-2d    %-12d    %-10.2f    %-10.2f %s\n", count, side, mWorldSize, mCellSize, stepCount, calcTime, speed, inputFile);

		 fclose(out);*/

		char statFile[250];
		Utils::copyToLastChar(statFile, inputFile, '/', 2);

		sprintf(statFile, "%s%s", statFile, "statistic");

		char tempFile[250];
		Utils::copyToLastChar(tempFile, inputFile, '/', 2);

		sprintf(tempFile, "%s%s", tempFile, "tempFile");

		FILE* out;
		out = fopen(statFile, "a");

		FILE* tmp;
		tmp = fopen(tempFile, "a");

		int side = (int) sqrt(((double) count) / mCellSize);

		char in[50];
		Utils::copyFromLastToEnd(in, inputFile, '/', 2);

		fprintf(out, "Element count: %d\n"
				"Side (square): %d\n"
				"Thread count:  %d\n"
				"Cell size:     %d\n"
				"Step count:    %d\n"
				"Calc time:     %.2f\n"
				"Speed:         %.2f\n\n\n\n", count, side, mWorkerCommSize, mCellSize, stepCount, calcTime, speed);

		fclose(out);

		fprintf(tmp, "%d %6d %10.2f %10.2f\n", mWorkerCommSize, side, speed, calcTime);
		fclose(tmp);
	}

	return;
	/*if ( flags & STATISTICS ) {
	 if( world_rank == 0 ) {
	 int countGridNodes = getCountGridNodes();
	 int repeatCount = getRepeatCount();
	 double speed = (double)(countGridNodes) * repeatCount / calcTime / 1000000;

	 int* devices = new int[world_size * 2];
	 devices[0] = getCpuBlocksCount();
	 devices[1] = getGpuBlocksCount();

	 for (int i = 1; i < world_size; ++i) {
	 MPI_Recv(devices + 2 * i, 1, MPI_INT, i, 999, MPI_COMM_WORLD, &status);
	 MPI_Recv(devices + 2 * i + 1, 1, MPI_INT, i, 999, MPI_COMM_WORLD, &status);
	 }

	 ofstream out;
	 out.open(statisticsFile, ios::app);

	 out << "############################################################" << endl;
	 out.precision(5);
	 out << endl <<
	 "Input file:   " << inputFile << endl <<
	 "Output file:  " << outputFile << endl <<
	 "Node count:   " << countGridNodes << endl <<
	 "Repeat count: " << repeatCount << endl <<
	 "Time:         " << calcTime << endl <<
	 "Speed (10^6): " << speed << endl <<
	 endl;

	 for (int i = 0; i < world_size; ++i)
	 out << "Thread #" << i << " CPU blocks: " << devices[2 * i] << " GPU blocks: " << devices[2 * i + 1] << endl << endl;

	 out << "############################################################" << endl;

	 out.close();

	 delete devices;
	 }
	 else {
	 int cpuCount = getCpuBlocksCount();
	 int gpuCount = getGpuBlocksCount();

	 MPI_Send(&cpuCount, 1, MPI_INT, 0, 999, MPI_COMM_WORLD);
	 MPI_Send(&gpuCount, 1, MPI_INT, 0, 999, MPI_COMM_WORLD);
	 }
	 }*/
}

bool Domain::isNan() {
	bool flag = false;

	for (int i = 0; i < mBlockCount; ++i) {
		if (mBlocks[i]->isNan()) {
			flag = true;
		}
	}

	return flag;
}

void Domain::checkOptions(int flags, double _stopTime, char* saveFile) {
	if (flags & TIME_EXECUTION)
		setStopTime(_stopTime);

	if (flags & LOAD_FILE)
		loadStateFromFile(saveFile);
}

void Domain::createProcessigUnit() {
	gpu = new ProcessingUnit*[mGpuCount];

	switch (dimension) {
		case 1:
			cpu = new CPU_1d(0);

			for (int i = 0; i < mGpuCount; ++i) {
				gpu[i] = new GPU_1d(i);
			}

			break;
		case 2:
			cpu = new CPU_2d(0);

			for (int i = 0; i < mGpuCount; ++i) {
				gpu[i] = new GPU_2d(i);
			}

			break;
		case 3:
			cpu = new CPU_3d(0);

			for (int i = 0; i < mGpuCount; ++i) {
				gpu[i] = new GPU_3d(i);
			}

			break;
		default:
			break;
	}
}

void Domain::createBlock(ifstream& in) {
	readBlockCount(in);

	mBlocks = new Block*[mBlockCount];

	for (int i = 0; i < mBlockCount; ++i)
		mBlocks[i] = readBlock(in, i, dimension);
}

void Domain::createInterconnect(ifstream& in) {
	readConnectionCount(in);

	mInterconnects = new Interconnect*[mConnectionCount];

	for (int i = 0; i < mConnectionCount; ++i)
		mInterconnects[i] = readConnection(in);
}

void Domain::initSolverInfo() {
	switch (mSolverIndex) {
		case EULER:
			mSolverInfo = new EulerStorage();
			break;
		case RK4:
			mSolverInfo = new RK4Storage();
			break;
		case DP45:
			mSolverInfo = new DP45Storage();
			break;
		default:
			mSolverInfo = new EulerStorage();
			break;
	}
}

void Domain::blockAfterCreate() {
	for (int i = 0; i < mBlockCount; ++i) {
		mBlocks[i]->afterCreate(mProblenType, mSolverIndex, mAtol, mRtol);
	}
}

Interconnect* Domain::getInterconnect(int sourceNode, int destinationNode, int borderLength, double* sourceData,
		double* destinationData) {
	if (sourceNode == destinationNode)
		return new NonTransferInterconnect(sourceNode, destinationNode);

	if (mWorkerRank == sourceNode) {
		return new TransferInterconnectSend(sourceNode, destinationNode, borderLength, sourceData, &mWorkerComm);
	}

	if (mWorkerRank == destinationNode) {
		return new TransferInterconnectRecv(sourceNode, destinationNode, borderLength, destinationData, &mWorkerComm);;
	}

	return new NonTransferInterconnect(sourceNode, destinationNode);
}

int Domain::getMaxStepStorageCount() {
	/*int cpuElementCount = 0;
	 for (int i = 0; i < mBlockCount; ++i) {
	 if (mBlocks[i]->isProcessingUnitCPU()) {
	 cpuElementCount += mBlocks[i]->getGridElementCount();
	 }
	 }

	 int gpu0ElementCount = 0;
	 for (int i = 0; i < mBlockCount; ++i) {
	 if (mBlocks[i]->isProcessingUnitGPU() && mBlocks[i]->getDeviceNumber() == 0) {
	 gpu0ElementCount += mBlocks[i]->getGridElementCount();
	 }
	 }

	 int gpu1ElementCount = 0;
	 for (int i = 0; i < mBlockCount; ++i) {
	 if (mBlocks[i]->isProcessingUnitGPU() && mBlocks[i]->getDeviceNumber() == 1) {
	 gpu1ElementCount += mBlocks[i]->getGridElementCount();
	 }
	 }

	 int gpu2ElementCount = 0;
	 for (int i = 0; i < mBlockCount; ++i) {
	 if (mBlocks[i]->isProcessingUnitGPU() && mBlocks[i]->getDeviceNumber() == 2) {
	 gpu2ElementCount += mBlocks[i]->getGridElementCount();
	 }
	 }

	 int solverSizeCpu = mSolverInfo->getSize(cpuElementCount);
	 int solverSizeGpu0 = mSolverInfo->getSize(gpu0ElementCount);
	 int solverSizeGpu1 = mSolverInfo->getSize(gpu1ElementCount);
	 int solverSizeGpu2 = mSolverInfo->getSize(gpu2ElementCount);

	 int maxCountCpu = CPU_RAM / solverSizeCpu;
	 int maxCountGpu0 = GPU_RAM / solverSizeGpu0;
	 int maxCountGpu1 = GPU_RAM / solverSizeGpu1;
	 int maxCountGpu2 = GPU_RAM / solverSizeGpu2;

	 int maxCount = min(maxCountCpu, min(maxCountGpu0, min(maxCountGpu1, maxCountGpu2)));

	 return maxCount;*/

	int cpuElementCount = 0;
	int* gpuElementCount = new int[mGpuCount];
	for (int i = 0; i < mGpuCount; ++i) {
		gpuElementCount[i] = 0;
	}

	cpuElementCount = getElementCountOnProcessingUnit(CPUNIT, 0);

	for (int i = 0; i < mGpuCount; ++i) {
		gpuElementCount[i] = getElementCountOnProcessingUnit(GPUNIT, i);
	}

	int cpuSolverSize = mSolverInfo->getSize(cpuElementCount);
	int* gpuSolverSize = new int[mGpuCount];
	for (int i = 0; i < mGpuCount; ++i) {
		gpuSolverSize[i] = mSolverInfo->getSize(gpuElementCount[i]);
	}

	int cpuMaxCount = (int) (CPU_RAM / cpuSolverSize);
	int* gpuMaxCount = new int[mGpuCount];
	for (int i = 0; i < mGpuCount; ++i) {
		gpuMaxCount[i] = (int) (GPU_RAM / gpuSolverSize[i]);
	}

	int minForAll = cpuMaxCount;
	for (int i = 0; i < mGpuCount; ++i) {
		if (gpuMaxCount[i] < minForAll)
			minForAll = gpuMaxCount[i];
	}

	int absMin;

	MPI_Allreduce(&minForAll, &absMin, 1, MPI_INT, MPI_MIN, mWorkerComm);

	delete gpuElementCount;
	delete gpuSolverSize;
	delete gpuMaxCount;

	return absMin;
}

int Domain::getElementCountOnProcessingUnit(int deviceType, int deviceNumber) {
	int count = 0;
	for (int i = 0; i < mBlockCount; ++i) {
		if (mBlocks[i]->isBlockType(deviceType) && mBlocks[i]->isDeviceNumber(deviceNumber)) {
			count += mBlocks[i]->getGridElementCount();
		}
	}

	return count;
}

