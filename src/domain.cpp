/*
 * Domain.cpp
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#include "domain.h"

#include <cassert>
#include <stdlib.h>

#include <stdio.h>
#include <iostream>

using namespace std;

Domain::Domain(int _world_rank, int _world_size, char* inputFile, char* binaryFileName, int _jobId) {
	mGlobalRank = _world_rank;
	MPI_Comm_split(MPI_COMM_WORLD, 1, _world_rank, &mWorkerComm);

	MPI_Comm_size(mWorkerComm, &mWorkerCommSize);
	MPI_Comm_rank(mWorkerComm, &mWorkerRank);

	if (mWorkerCommSize != _world_size) {
		printwcts("Communicator size error!", LL_DEBUG);
		assert(0);
	}

	mJobId = _jobId;
	Utils::getTracerFolder(binaryFileName, mTracerFolder);
	Utils::getProjectFolder(inputFile, mProjectFolder);
	currentTime = 0;
	mStepCount = 0;

	mTimeStep = 0;
	stopTime = 0;

	mRepeatCount = 0;

	mSaveTimer = mSavePeriod;

	dimension = 0;

	cpu = NULL;
	gpu = NULL;

	mGpuCount = GPU_COUNT;

	printwcts("PROBLEM TYPE ALWAYS = ORDINARY!!!\n", LL_DEBUG);
	mProblemType = ORDINARY;

	readFromFile(inputFile);

	mAcceptedStepCount = 0;
	mRejectedStepCount = 0;

	isRealTimePNG = true;
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

	if (mPlotPeriods)
		delete mPlotPeriods;

	if (mPlotTimers)
		delete mPlotTimers;

}

int Domain::getUserStatus() {
	if (mJobId < 0)
		return US_RUN;
	else
		//TODO implement http response for current jobId
		printwcts("user status getter not implemented", LL_INFO);
	assert(0);
	return US_RUN;
}

int Domain::isReadyToFullSave() {
	int result = 0;
	if (mSavePeriod > 0) {
		mSaveTimer -= mTimeStep;
		if (mSaveTimer <= 0)
			result = 1;
		while (mSaveTimer <= 0)
			mSaveTimer += mSavePeriod;
	}
	return result;
}

int Domain::isReadyToPlot() {
	//here we subtract mTimeStep from every timer and
	//return 1 if any of them is below or equal to 0
	//then we add sufficient amount of periods to make it positive again
	//result is sum_n 2^(n+1)*(1 if nth plot is needed else 0)
	int result = 0;
	for (int i = mPlotCount - 1; i >= 0; i--) {
		if (mPlotPeriods[i] > 0) {
			mPlotTimers[i] -= mTimeStep;
			if (mPlotTimers[i] <= 0)
				result += 1;
			while (mPlotTimers[i] <= 0)
				mPlotTimers[i] += mPlotPeriods[i];
		}
		result *= 2;
	}
	return result;
}

void Domain::stopByUser(char* inputFile) {
	mJobState = JS_CANCELLED;
	saveStateForLoad(inputFile);
}

void Domain::stopByTime(char* inputFile) {
	printwcts("\nSave the final result to a file. The procedure may take time.\n", LL_INFO);
	saveStateForLoad(inputFile);

	double theta = getThetaForDenseOutput(stopTime);
	currentTime = stopTime;
	saveStateForDrawDenseOutput(inputFile, theta);
}

int Domain::getEntirePlotValues() {
	//returns code for every possible plot
	int result = 0;
	for (int i = 0; i < mPlotCount; i++) {
		result += 1;
		result *= 2;
	}
	return result;
}

void Domain::compute(char* inputFile) {
	double wnow = MPI_Wtime();

	printwcts("Running computations mpi rank %d \n", LL_INFO);

	printwcts("Tracer root folder: " + ToString(mTracerFolder) + "\n", LL_INFO);
	printwcts("Project folder: " + ToString(mProjectFolder) + "\n", LL_INFO);
	printwcts(
			"Computing from " + ToString(currentTime) + " to " + ToString(stopTime) + " with step "
					+ ToString(mTimeStep) + "\n", LL_INFO);
	printwcts("Computation started, worker #" + ToString(mWorkerRank) + "\n", LL_INFO);
	printwcts("solver stage count: " + ToString(mNumericalMethod->getStageCount()) + "\n", LL_INFO);

	if (mNumericalMethod->isFSAL())
		initSolvers();

	double computeInterval = stopTime - currentTime;
	int percentage = 0;

	//1.
	mUserStatus = US_RUN;
	mJobState = JS_RUNNING;

	if (mWorkerRank == 0) {
		mUserStatus = getUserStatus();
		printwcts("Initial user status received: " + ToString(mUserStatus) + "\n", LL_INFO);
	}
	MPI_Bcast(&mUserStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
	// TODO если пользователь остановил расчеты, то необходимо выполнить сохранение для загузки состояния (saveStateForLoad)

	while ((mUserStatus != US_STOP) && (mJobState == JS_RUNNING)) {
		nextStep();

		//printBlocksToConsole();
		int newPercentage = 100.0 * (1.0 - (stopTime - currentTime) / computeInterval);
		int percentChanged = newPercentage > percentage;

		if (percentChanged) {
			double wnow2 = MPI_Wtime();
			percentage = newPercentage;
			if (mWorkerRank == 0) {
				//printwcts("Done " + ToString(percentage) + "% in " + ToString((int) (now2-now))+ " seconds, and wtime gives " + ToString((wnow2-wnow))+ " seconds, and omp_wtime gives " + ToString((mnow2-mnow))+ " seconds, ratio is "  + ToString((wnow2-wnow)/(mnow2-mnow)) + " \n", LL_INFO);
				printwcts(
						"Done " + ToString(percentage) + "% in " + ToString((wnow2 - wnow)) + " seconds, ETA = "
								+ ToString((100 - percentage) * (wnow2 - wnow)) + " seconds\n", LL_INFO);
				//printwcts("Done " + ToString(percentage) + "% in " + ToString((mnow2-mnow))+ " seconds, and wtime gives " + ToString((wnow2-wnow))+ " seconds, ETA = "+ ToString((100-percentage)*(mnow2-mnow)) + " seconds\n" , LL_INFO);
			}
			wnow = wnow2;
		}

		/*if (!(currentTime < stopTime)) {
		 mJobState = JS_FINISHED;
		 }*/

		if (isReadyToFullSave()) {
			saveStateForLoad(inputFile);
		}

		int plotVals = isReadyToPlot();
		if (plotVals) {
			saveStateForDraw(inputFile, plotVals);
		}

		//check for termination request
		if (mWorkerRank == 0) {
			mUserStatus = getUserStatus();
		}
		MPI_Bcast(&mUserStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);

		if (mJobState == JS_FINISHED) {
			stopByTime(inputFile);
		}
	} //end while

	printwcts("Computation finished for worker #" + ToString(mWorkerRank) + "\n", LL_INFO);

}

void Domain::initSolvers() {
	computeStage(SOLVER_INIT_STAGE);
}

void Domain::computeStage(int stage) {
	// TODO: Проверить порядок следующих 4-х функций
	mProblem->computeStageData(currentTime, mTimeStep, mNumericalMethod->getStageTimeStepCoefficient(stage));

	prepareStageArgument(stage);

	prepareBlockStageData(stage);

	prepareData(stage);

	for (int i = 0; i < mConnectionCount; ++i)
		mInterconnects[i]->transfer();

	computeOneStepCenter(stage);

	for (int i = 0; i < mConnectionCount; ++i)
		mInterconnects[i]->wait();

	computeOneStepBorder(stage);

	//prepareNextStageArgument(stage);

	/*printf("\nstage #%d\n", stage);
	 printBlocksToConsole();*/
}

void Domain::nextStep() {
	//int totalGridElements = getGridElementCount();
	//последовательно выполняем все стадии метода
	for (int stage = 0; stage < mNumericalMethod->getStageCount(); stage++)
		computeStage(stage);

	//!!! Собрать мастеру ошибки
	//!!! если ошибки нет, продолжать
	double error = 0.0;
	bool isErrorPer = true;
	if (mNumericalMethod->isVariableStep()) {
		// MPI inside!
		error = collectError();
		//printwcts("step error = " + ToString(error) + " " + ToString(currentTime) + "\n", LL_INFO);
		isErrorPer = isErrorPermissible(error);

		/*		if (!isErrorPer) {
		 //printwcts(ToString(currentTime) +" " + ToString(mTimeStep) + " " + ToString(stopTime) + "\n", LL_INFO);
		 if (currentTime + mTimeStep > stopTime) {
		 mUserStatus = US_STOP;
		 return;
		 }

		 //printwcts("###\n", LL_INFO);

		 confirmStep();
		 mAcceptedStepCount++;
		 currentTime += mTimeStep;
		 //cout<<"Step accepted\n"<<endl;
		 } else {
		 rejectStep();
		 mRejectedStepCount++;
		 //cout << "Step rejected!\n" << endl;

		 mTimeStep = computeNewStep(error);
		 printwcts("new time step = " + ToString(mTimeStep) + "\n", LL_INFO);
		 if(mTimeStep < MINIMALLY_ACCEPTABLE_TIMESTEP) {
		 printwcts("New time step (" + ToString(mTimeStep) + ") too small. Computation aborted.\n\n", LL_INFO);
		 mJobState = JS_FINISHED;
		 }

		 return;
		 }*/
	}

	if (isErrorPer) {
		if (currentTime + mTimeStep > stopTime) {
			mJobState = JS_FINISHED;
			return;
		}

		confirmStep();
		currentTime += mTimeStep;
		mAcceptedStepCount++;
	} else {
		rejectStep();
		mRejectedStepCount++;
	}

	if (mNumericalMethod->isVariableStep()) {
		mTimeStep = computeNewStep(error);
		//printwcts("new time step = " + ToString(mTimeStep) + "\n", LL_INFO);
		if (mTimeStep < MINIMALLY_ACCEPTABLE_TIMESTEP) {
			printwcts("New time step (" + ToString(mTimeStep) + ") too small. Computation aborted.\n\n", LL_INFO);
			mJobState = JS_FINISHED;
		}
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
			mBlocks[i]->computeStageBorder(stage,
					currentTime + mNumericalMethod->getStageTimeStepCoefficient(stage) * mTimeStep);
		}
}

void Domain::processDeviceBlocksCenter(int deviceType, int deviceNumber, int stage) {
	for (int i = 0; i < mBlockCount; ++i)
		if (mBlocks[i]->isBlockType(deviceType) && mBlocks[i]->isDeviceNumber(deviceNumber)) {
			//cout << endl << "ERROR! PROCESS DEVICE!" << endl;
			mBlocks[i]->computeStageCenter(stage,
					currentTime + mNumericalMethod->getStageTimeStepCoefficient(stage) * mTimeStep);
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

void Domain::prepareStageArgument(int stage) {
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

void Domain::prepareBlockStageData(int stage) {
	for (int i = 0; i < mBlockCount; ++i) {
		mBlocks[i]->prepareStageSourceResult(stage, mTimeStep, currentTime);
	}
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
	mProblem->confirmStep(currentTime);
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
		isErrorPermissible = (int) (mNumericalMethod->isErrorPermissible(error, totalGridElementCount));
	}

	MPI_Bcast(&isErrorPermissible, 1, MPI_INT, 0, mWorkerComm);
	return (bool) (isErrorPermissible);
}

double Domain::computeNewStep(double error) {
	double newStep = 0;

	if (mWorkerRank == 0) {
		newStep = mNumericalMethod->computeNewStep(mTimeStep, error, totalGridElementCount);
	}

	MPI_Bcast(&newStep, 1, MPI_DOUBLE, 0, mWorkerComm);
	return newStep; //mNumericalMethod->computeNewStep(mTimeStep, error, totalGridElementCount);
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
	readSavePeriod(in);
	readGridSteps(in);

	in.read((char*) &dimension, SIZE_INT);
	createProcessigUnit();

	readCellAndHaloSize(in);
	readSolverIndex(in);
	readSolverTolerance(in);

	createNumericalMethod();

	/*mProblemType = DELAY;
	 int stateCount = 101000;
	 int delayCount  = 1;
	 double* delayValue = new double[1];
	 delayValue[0] = 1.0;
	 createProblem(stateCount, delayCount, delayValue);
	 delete delayValue;*/
	readProblem(in);

	createBlock(in);

	createInterconnect(in);

	//todo send every left side of interconnect to

	readPlots(in);

	for (int i = 0; i < mBlockCount; ++i)
		mBlocks[i]->moveTempBorderVectorToBorderArray();

	totalGridNodeCount = getGridNodeCount();
	totalGridElementCount = getGridElementCount();

	//blockAfterCreate();

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

void Domain::readSavePeriod(ifstream& in) {
	in.read((char*) &mSavePeriod, SIZE_DOUBLE);

	//cout << "save interval: " << savePeriod << endl;
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

void Domain::readProblem(std::ifstream& in) {
	/*mProblemType = DELAY;
	 int stateCount = 101000;
	 int delayCount  = 1;
	 double* delayValue = new double[1];
	 delayValue[0] = 1.0;
	 createProblem(stateCount, delayCount, delayValue);
	 delete delayValue;*/

	in.read((char*) &mProblemType, SIZE_INT);

	if (mProblemType == 0) {
		mProblemType = ORDINARY;
		mProblem = new OrdinaryProblem();
	}

	if (mProblemType == 1) {
		mProblemType = DELAY;

		int stateCount = 101000;
		int delayCount = 0;
		double* delayValue = NULL;

		in.read((char*) &delayCount, SIZE_INT);
		delayValue = new double[delayCount];
		for (int i = 0; i < delayCount; ++i) {
			in.read((char*) &delayValue[i], SIZE_DOUBLE);
		}
		mProblem = new DelayProblem(stateCount, delayCount, delayValue);
		delete delayValue;
	}
}

void Domain::readBlockCount(ifstream& in) {
	in.read((char*) &mBlockCount, SIZE_INT);

	//cout << "block count:   " << mBlockCount << endl;
}

void Domain::readPlotCount(ifstream& in) {
	in.read((char*) &mPlotCount, SIZE_INT);

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
					printwcts("Invalid block device number for CPU!\n", LL_INFO);
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
				printwcts("Invalid block device number for GPU!\n", LL_INFO);
				assert(false);
			}

			pu = gpu[deviceNumber];
		} else {
			printwcts("Invalid block type!\n", LL_INFO);
			assert(false);
		}

		resBlock = new RealBlock(node, dimension, count[0], count[1], count[2], mCellSize, mHaloSize, idx, pu,
				initFuncNumber, compFuncNumber, mProblem, mNumericalMethod);
	} else {
		//resBlock =  new BlockNull(idx, dimension, count[0], count[1], count[2], offset[0], offset[1], offset[2], node, deviceNumber, mHaloSize, mCellSize);
		resBlock = new NullBlock(node, dimension, count[0], count[1], count[2], mCellSize, mHaloSize);
	}

	delete initFuncNumber;
	delete compFuncNumber;

	//resBlock->createSolver(mSolverIndex, mAtol, mRtol);

	return resBlock;
}

/*
 *  исправление границ, принадлежащих склейке
 *  если мы на блоке-получателе со старшей стороны, нужно получить начальные условия из соседнего блока
 */
void Domain::fixInitialBorderValues(int sourceBlock, int destinationBlock, int* offsetSource, int* offsetDestination,
		int* length, int sourceSide, int destinationSide) {
	//printf("welcome to border fixer\n");
	int sourceNode = mBlocks[sourceBlock]->getNodeNumber();
	int destinationNode = mBlocks[destinationBlock]->getNodeNumber();
	//of the two halves of interconnect we use [--) <-- [--]  and ignore the other
	if ((destinationSide == RIGHT) or (destinationSide == BACK) or (destinationSide == BOTTOM)) {
		//printf("fixing side %d \n", destinationSide);
		double* destBuffer = NULL;
		double* sourceBuffer = NULL;
		int bufferSize = mCellSize * length[0] * length[1];
		ProcessingUnit* sourcePU = NULL;
		ProcessingUnit* destPU = NULL;
		MPI_Request request;

		if (mWorkerRank == sourceNode) {
			//готовь источник
			sourcePU = mBlocks[sourceBlock]->getPU();
			sourceBuffer = sourcePU->newDoubleArray(bufferSize);
			//заполняй источник
			int smStart = offsetSource[0];
			int snStart = offsetSource[1];
			int smStop = offsetSource[0] + length[0];
			int snStop = offsetSource[1] + length[1];
			//printf("copying %d %d %d %d \n",smStart, smStop, snStart, snStop);
			mBlocks[sourceBlock]->getSubVolume(sourceBuffer, smStart, smStop, snStart, snStop, sourceSide);
			//printf("got subvolume of size %d\n", bufferSize);
			//for (int i =0; i< bufferSize; i++)
			//printf("%f ", sourceBuffer[i]);
			//printf("\n");
			//printf("sending to %d...\n", destinationNode);
			MPI_Isend(sourceBuffer, bufferSize, MPI_DOUBLE, destinationNode, 0, mWorkerComm, &request);
			//printf("Isend ok\n");
		}
		if (mWorkerRank == destinationNode) {
			//готовь дестинэйшн
			destPU = mBlocks[destinationBlock]->getPU();
			destBuffer = destPU->newDoubleArray(bufferSize);
			//копируй дестинэйшн в блок
			MPI_Status mpistatus;
			//printf("receiving buffer\n");
			MPI_Recv(destBuffer, bufferSize, MPI_DOUBLE, sourceNode, 0, mWorkerComm, &mpistatus);
			//printf("receive ok\n");
			int dmStart = offsetDestination[0];
			int dnStart = offsetDestination[1];
			int dmStop = offsetDestination[0] + length[0];
			int dnStop = offsetDestination[1] + length[1];
			mBlocks[destinationBlock]->setSubVolume(destBuffer, dmStart, dmStop, dnStart, dnStop, destinationSide);
		}

		if (mWorkerRank == sourceNode) {
			MPI_Status mpistatus;
			MPI_Wait(&request, &mpistatus);
			sourcePU->deleteDeviceSpecificArray(sourceBuffer);
		}
		if (mWorkerRank == destinationNode) {
			destPU->deleteDeviceSpecificArray(destBuffer);
		}
	}

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

	fixInitialBorderValues(sourceBlock, destinationBlock, offsetSource, offsetDestination, length,
			Utils::getSide(sourceSide), Utils::getSide(destinationSide));

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

void Domain::saveStateForDraw(char* inputFile, int plotVals) {
	char* saveFile = new char[250];
	Utils::getFilePathForDraw(inputFile, saveFile, currentTime, plotVals);
	printwcts("produced results for t=" + ToString(currentTime) + ": " + ToString(saveFile) + "\n", LL_INFO);
	saveGeneralInfo(saveFile);
	saveStateForDrawByBlocks(saveFile);

	if (isRealTimePNG) {
		char comline[250];
		sprintf(comline, "python %s/hybriddomain/plotter.py %s", mTracerFolder, saveFile);
		printwcts("comm line = " + ToString(comline) + "\n", LL_INFO);
		system(comline);
	}

	delete saveFile;
}

void Domain::saveStateForLoad(char* inputFile) {
	char* saveFile = new char[250];
	Utils::getFilePathForLoad(inputFile, saveFile, currentTime);

	saveGeneralInfo(saveFile);
	saveProblem(saveFile);
	saveStateForLoadByBlocks(saveFile);

	delete saveFile;
}

void Domain::saveStateForDrawDenseOutput(char* inputFile, double theta) {
	char* saveFile = new char[250];
	Utils::getFilePathForDraw(inputFile, saveFile, stopTime, getEntirePlotValues());

	saveGeneralInfo(saveFile);
	saveStateForDrawDenseOutputByBlocks(saveFile, theta);

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

void Domain::saveProblem(char* path) {
	if (mGlobalRank == 0) {
		mProblem->save(path);
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

void Domain::saveStateForDrawDenseOutputByBlocks(char* path, double theta) {
	for (int i = 0; i < mBlockCount; ++i) {
		mBlocks[i]->saveStateForDrawDenseOutput(path, mTimeStep, theta);
		MPI_Barrier(mWorkerComm);
	}
}

double Domain::getThetaForDenseOutput(double requiredTime) {
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

	mProblem->load(in);

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

		printwcts("Steps accepted: " + ToString(mAcceptedStepCount) + "\n", LL_INFO);
		printwcts("Steps rejected: " + ToString(mRejectedStepCount) + "\n", LL_INFO);
		printwcts("Time: " + ToString(calcTime) + "\n", LL_INFO);
		printwcts("Element count: " + ToString(count) + "\n", LL_INFO);
		printwcts("Performance (10^6): " + ToString(speed) + "\n\n", LL_INFO);

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

	if (flags & NO_REALTIME_PNG)
		isRealTimePNG = false;
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

void Domain::readPlots(ifstream& in) {
	readPlotCount(in);

	mPlotPeriods = new double[mPlotCount];
	mPlotTimers = new double[mPlotCount];
	in.read((char*) mPlotPeriods, mPlotCount * SIZE_DOUBLE);
	for (int i = 0; i < mPlotCount; i++)
		mPlotTimers[i] = mPlotPeriods[i];

	printwcts("read plots: " + ToString(mPlotCount) + "\n", LL_INFO);
	for (int i = 0; i < mPlotCount; i++)
		printwcts("plot: " + ToString(mPlotPeriods[i]) + "\n", LL_INFO);
}

void Domain::createInterconnect(ifstream& in) {
	readConnectionCount(in);

	mInterconnects = new Interconnect*[mConnectionCount];

	for (int i = 0; i < mConnectionCount; ++i)
		mInterconnects[i] = readConnection(in);
}

void Domain::createNumericalMethod() {
	switch (mSolverIndex) {
		case EULER:
			mNumericalMethod = new Euler(mAtol, mRtol);
			break;
		case RK4:
			mNumericalMethod = new RungeKutta4(mAtol, mRtol);
			break;
		case DP45:
			mNumericalMethod = new DormandPrince45(mAtol, mRtol);
			break;
		default:
			mNumericalMethod = new Euler(mAtol, mRtol);
			break;
	}
}

void Domain::createProblem(int stateCount, int delayCount, double* delayValue) {
	if (mProblemType == ORDINARY) {
		mProblem = new OrdinaryProblem();
		return;
	}

	if (mProblemType == DELAY) {
		mProblem = new DelayProblem(stateCount, delayCount, delayValue);
		return;
	}

	mProblem = new OrdinaryProblem();
}

void Domain::blockAfterCreate() {
	/*for (int i = 0; i < mBlockCount; ++i) {
	 mBlocks[i]->afterCreate(mProblenType, mSolverIndex, mAtol, mRtol);
	 }*/
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
	int cpuRequiredMemory = 0;
	int* gpuRequiredMemory = new int[mGpuCount];
	for (int i = 0; i < mGpuCount; ++i) {
		gpuRequiredMemory[i] = 0;
	}

	cpuRequiredMemory = getRequiredMemoryOnProcessingUnit(CPUNIT, 0);

	for (int i = 0; i < mGpuCount; ++i) {
		gpuRequiredMemory[i] = getRequiredMemoryOnProcessingUnit(GPUNIT, i);
	}

	/*int cpuSolverSize = mNumericalMethod->getMemorySizePerState(cpuRequiredMemory);
	 int* gpuSolverSize = new int[mGpuCount];
	 for (int i = 0; i < mGpuCount; ++i) {
	 gpuSolverSize[i] = mNumericalMethod->getMemorySizePerState(gpuRequiredMemory[i]);
	 }*/

	int cpuMaxCount = (int) (CPU_RAM / cpuRequiredMemory);
	int* gpuMaxCount = new int[mGpuCount];
	for (int i = 0; i < mGpuCount; ++i) {
		gpuMaxCount[i] = (int) (GPU_RAM / gpuRequiredMemory[i]);
	}

	int minForAll = cpuMaxCount;
	for (int i = 0; i < mGpuCount; ++i) {
		if (gpuMaxCount[i] < minForAll)
			minForAll = gpuMaxCount[i];
	}

	int absMin;

	MPI_Allreduce(&minForAll, &absMin, 1, MPI_INT, MPI_MIN, mWorkerComm);

	delete gpuRequiredMemory;
	//delete gpuSolverSize;
	delete gpuMaxCount;

	return absMin;
}

int Domain::getRequiredMemoryOnProcessingUnit(int deviceType, int deviceNumber) {
	int requiredMemory = 0;
	int elememtCount = 0;
	for (int i = 0; i < mBlockCount; ++i) {
		if (mBlocks[i]->isBlockType(deviceType) && mBlocks[i]->isDeviceNumber(deviceNumber)) {
			elememtCount += mBlocks[i]->getGridElementCount();
			requiredMemory += mNumericalMethod->getMemorySizePerState(elememtCount);
		}
	}

	return requiredMemory;
}

