/*
 * Domain.cpp
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#include "domain.h"
#include "solvers/solver.h"
#include <cassert>

using namespace std;

int lastChar(char* source, char ch) {
	int i = 0;
	int index= 0;
	while(source[i] != 0) {
		if(source[i] == ch)
			index = i;
		i++;
	}

	return index;
}

Domain::Domain(int _world_rank, int _world_size, char* inputFile, int _flags, int _stepCount, double _stopTime, char* loadFile) {
	mWorldRank = _world_rank;
	mWorldSize = _world_size;

	currentTime = 0;
	mStepCount = 0;

	timeStep = 0;
	stopTime = 0;

	mRepeatCount = 0;

	counterSaveTime = 0;

	readFromFile(inputFile);

	/*flags = _flags;

	if( flags & LOAD_FILE )
		loadStateFromFile(inputFile, loadFile);
	else
		readFromFile(inputFile);

	if( flags & TIME_EXECUTION )
		stopTime = _stopTime;

	if( flags & STEP_EXECUTION )
		mStepCount = _stepCount;*/

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
}

double** Domain::collectDataFromNode() {
	double** resultAll = NULL;

	if(mWorldRank == 0) {
		resultAll = new double* [mBlockCount];
	}

	for (int i = 0; i < mBlockCount; ++i) {
		double* tmp = getBlockCurrentState(i);

		if(mWorldRank == 0)
			resultAll[i] = tmp;
	}

	return resultAll;
}

double* Domain::getBlockCurrentState(int number) {
	double* result = new double [mBlocks[number]->getGridElementCount()];

	if(mWorldRank == 0) {
		if(mBlocks[number]->isRealBlock()) {
			mBlocks[number]->getCurrentState(result);
		}
		else {
			MPI_Recv(result, mBlocks[number]->getGridElementCount(), MPI_DOUBLE, mBlocks[number]->getNodeNumber(), 999, MPI_COMM_WORLD, &status);
		}

		return result;
	}
	else {
		if(mBlocks[number]->isRealBlock()) {
			mBlocks[number]->getCurrentState(result);
			MPI_Send(result, mBlocks[number]->getGridElementCount(), MPI_DOUBLE, 0, 999, MPI_COMM_WORLD);
			delete result;
			return NULL;
		}
	}
	return NULL;
}



void Domain::compute(char* inputFile) {
	cout << endl << "Computation started..." << endl;
	cout << "Current time: "<<currentTime<<", finish time: "<<stopTime<< ", time step: " << timeStep<<endl;
	cout << "solver stage count: " << mSolverInfo->getStageCount() << endl;

	if (mSolverInfo->isFSAL() )
	    initSolvers();

	/*if( flags & STEP_EXECUTION)
		for (int i = 0; i < mStepCount; i++)
			nextStep();
	else*/
	while ( currentTime < stopTime ){
		nextStep();

		if( saveInterval != 0 ) {
			counterSaveTime += timeStep;

			if( counterSaveTime > saveInterval ) {
				counterSaveTime = 0;
				saveState(inputFile);
			}
		}
	}
	cout <<"Computation finished!" << endl;
}

void Domain::initSolvers() {
	computeStage(SOLVER_INIT_STAGE);
}

void Domain::computeStage(int stage) {
	prepareData(stage);

	for (int i = 0; i < mConnectionCount; ++i)
		mInterconnects[i]->sendRecv(mWorldRank);

	computeOneStepCenter(stage);

	for (int i = 0; i < mConnectionCount; ++i)
		mInterconnects[i]->wait();

	computeOneStepBorder(stage);
	prepareNextStageArgument(stage);
}


void Domain::nextStep() {
	//int totalGridElements = getGridElementCount();
	//последовательно выполняем все стадии метода
    for(int stage=0; stage < mSolverInfo->getStageCount(); stage++)
    	computeStage(stage);

    if (mSolverInfo->isVariableStep()){
    	double error = collectError();
    	printf("step error = %f\n", error);
    	timeStep = mSolverInfo->getNewStep(timeStep, error,totalGridElementCount);
		if (mSolverInfo->isErrorPermissible(error,totalGridElementCount)){
			confirmStep(); //uses new timestep
			mAcceptedStepCount++;
			currentTime += timeStep;
			//cout<<"Step accepted\n"<<endl;
		}
		else{
			rejectStep(); //uses new timestep
			mRejectedStepCount++;
			cout<<"Step rejected!\n"<<endl;
		}

		printf("new time step = %f\n", timeStep);
    }
	else{ //constant step
		confirmStep();
		mAcceptedStepCount++;
		currentTime += timeStep;

	}
}

void Domain::prepareDeviceData(int deviceType, int deviceNumber, int stage) {
	for (int i = 0; i < mBlockCount; ++i)
		if( mBlocks[i]->getBlockType() == deviceType && mBlocks[i]->getDeviceNumber() == deviceNumber ) {
			mBlocks[i]->prepareStageData(stage);
		}
}

void Domain::processDeviceBlocksBorder(int deviceType, int deviceNumber, int stage) {
	for (int i = 0; i < mBlockCount; ++i)
        if( mBlocks[i]->getBlockType() == deviceType && mBlocks[i]->getDeviceNumber() == deviceNumber ) {
        	//cout << endl << "ERROR! PROCESS DEVICE!" << endl;
		    mBlocks[i]->computeStageBorder(stage, currentTime);
		}
}

void Domain::processDeviceBlocksCenter(int deviceType, int deviceNumber, int stage) {
	for (int i = 0; i < mBlockCount; ++i)
        if( mBlocks[i]->getBlockType() == deviceType && mBlocks[i]->getDeviceNumber() == deviceNumber ) {
        	//cout << endl << "ERROR! PROCESS DEVICE!" << endl;
		    mBlocks[i]->computeStageCenter(stage, currentTime);
		}
}
void Domain::prepareDeviceArgument(int deviceType, int deviceNumber, int stage) {
	for (int i = 0; i < mBlockCount; ++i)
        if( mBlocks[i]->getBlockType() == deviceType && mBlocks[i]->getDeviceNumber() == deviceNumber ) {
        	//cout << endl << "ERROR! PROCESS DEVICE!" << endl;
		    mBlocks[i]->prepareArgument(stage, timeStep);
		}
}

double Domain::getDeviceError(int deviceType, int deviceNumber) {
	double error=0;
	for (int i = 0; i < mBlockCount; ++i)
        if( mBlocks[i]->getBlockType() == deviceType && mBlocks[i]->getDeviceNumber() == deviceNumber ) {
        	//cout << endl << "ERROR! PROCESS DEVICE!" << endl;
		    error+=mBlocks[i]->getSolverStepError(timeStep);
		}
	return error;
}



void Domain::prepareData(int stage) {
#pragma omp task
	prepareDeviceData(GPU, 0, stage);
#pragma omp task
	prepareDeviceData(GPU, 1, stage);
#pragma omp task
	prepareDeviceData(GPU, 2, stage);

	prepareDeviceData(CPU, 0, stage);

#pragma omp taskwait
}

void Domain::computeOneStepBorder(int stage) {
#pragma omp task
	processDeviceBlocksBorder(GPU, 0, stage);
#pragma omp task
	processDeviceBlocksBorder(GPU, 1, stage);
#pragma omp task
	processDeviceBlocksBorder(GPU, 2, stage);

	processDeviceBlocksBorder(CPU, 0, stage);
}

void Domain::prepareNextStageArgument(int stage) {
#pragma omp task
	prepareDeviceArgument(GPU, 0, stage);
#pragma omp task
	prepareDeviceArgument(GPU, 1, stage);
#pragma omp task
	prepareDeviceArgument(GPU, 2, stage);

	prepareDeviceArgument(CPU, 0, stage);
}



void Domain::computeOneStepCenter(int stage) {
#pragma omp task
	processDeviceBlocksCenter(GPU, 0, stage);
#pragma omp task
	processDeviceBlocksCenter(GPU, 1, stage);
#pragma omp task
	processDeviceBlocksCenter(GPU, 2, stage);

	processDeviceBlocksCenter(CPU, 0, stage);
}


//TODO next two methods are not parallel!
void Domain::confirmStep() {
	for (int i = 0; i < mBlockCount; ++i) {
		mBlocks[i]->confirmStep(timeStep);
	}
}

void Domain::rejectStep() {
	for (int i = 0; i < mBlockCount; ++i) {
		mBlocks[i]->rejectStep(timeStep);
	}
}



double Domain::collectError() {
	double err1, err2, err3;
	err1=err2=err3=0;
    //1. Get total error for current node
#pragma omp task
	err1 = getDeviceError(GPU, 0);
#pragma omp task
	err2 = getDeviceError(GPU, 1);
#pragma omp task
	err3 = getDeviceError(GPU, 2);

	double nodeError = getDeviceError(CPU, 0);

#pragma omp taskwait
	nodeError += err1 + err2 + err3;

    //2. Collect errors from all nodes
    double totalError = nodeError;

	//TODO MPI_ALLREDUCE
	return totalError;
}

void Domain::printBlocksToConsole() {
	for (int i = 0; i < mBlockCount; ++i) {
		mBlocks[i]->print();
	}
}

void Domain::readFromFile(char* path) {
	ifstream in;
	in.open(path, ios::binary);

	readFileStat(in);
	readTimeSetting(in);
	readSaveInterval(in);
	readGridSteps(in);
	readCellAndHaloSize(in);
	readSolverIndex(in);
	readSolverTolerance(in);


	switch (mSolverIndex) {
		case EULER:
			mSolverInfo = new EulerSolver();
			break;
		case RK4:
			mSolverInfo = new RK4Solver();
			break;
		case DP45:
			mSolverInfo = new DP45Solver();
			break;
		default:
			mSolverInfo = new EulerSolver();
			break;
	}

	readBlockCount(in);

	mBlocks = new Block* [mBlockCount];

	printf ("DEBUG reading blocks.\n ");

	for (int i = 0; i < mBlockCount; ++i)
		mBlocks[i] = readBlock(in, i);

	printf ("DEBUG blocks read.\n ");

	readConnectionCount(in);

	mInterconnects = new Interconnect* [mConnectionCount];

	for (int i = 0; i < mConnectionCount; ++i)
		mInterconnects[i] = readConnection(in);


	for (int i = 0; i < mBlockCount; ++i)
		mBlocks[i]->moveTempBorderVectorToBorderArray();

	totalGridNodeCount = getGridNodeCount();
	totalGridElementCount = getGridElementCount();

	//printBlocksToConsole();
}

void Domain::readFileStat(ifstream& in) {
	char fileType;
	char versionMajor;
	char versionMinor;

	in.read((char*)&fileType, 1);
	in.read((char*)&versionMajor, 1);
	in.read((char*)&versionMinor, 1);

	cout << endl;
	cout << "file type:     " << (unsigned int)fileType << endl;
	cout << "version major: " << (unsigned int)versionMajor << endl;
	cout << "version minor: " << (unsigned int)versionMinor << endl;
}

void Domain::readTimeSetting(ifstream& in) {
	in.read((char*)&startTime, SIZE_DOUBLE);
	in.read((char*)&stopTime, SIZE_DOUBLE);
	in.read((char*)&timeStep, SIZE_DOUBLE);

	cout << "start time:    " << startTime << endl;
	cout << "stop time:     " << stopTime << endl;
	cout << "step time:     " << timeStep << endl;
}

void Domain::readSaveInterval(ifstream& in) {
	in.read((char*)&saveInterval, SIZE_DOUBLE);

	cout << "save interval: " << saveInterval << endl;
}

void Domain::readGridSteps(ifstream& in) {
	in.read((char*)&mDx, SIZE_DOUBLE);
	in.read((char*)&mDy, SIZE_DOUBLE);
	in.read((char*)&mDz, SIZE_DOUBLE);

	cout << "dx:            " << mDx << endl;
	cout << "dy:            " << mDy << endl;
	cout << "dz:            " << mDz << endl;
}

void Domain::readCellAndHaloSize(ifstream& in) {
	in.read((char*)&mCellSize, SIZE_INT);
	in.read((char*)&mHaloSize, SIZE_INT);

	cout << "cell size:     " << mCellSize << endl;
	cout << "halo size:     " << mHaloSize << endl;
}


void Domain::readSolverIndex(std::ifstream& in){
	in.read((char*)&mSolverIndex, SIZE_INT);
	cout << "Solver index:  " << mSolverIndex << endl;
}

void Domain::readSolverTolerance(std::ifstream& in){
	in.read((char*)&mAtol, SIZE_DOUBLE);
	cout << "Solver absolute tolerance:  " << mAtol << endl;
	in.read((char*)&mRtol, SIZE_DOUBLE);
	cout << "Solver relative tolerance:  " << mRtol << endl;
}



void Domain::readBlockCount(ifstream& in) {
	in.read((char*)&mBlockCount, SIZE_INT);

	cout << "block count:   " << mBlockCount << endl;
}

void Domain::readConnectionCount(ifstream& in) {
	in.read((char*)&mConnectionCount, SIZE_INT);

	cout << "connection count:   " << mConnectionCount << endl;
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
Block* Domain::readBlock(ifstream& in, int idx) {
	Block* resBlock;
	int dimension;
	int node;
	int deviceType;
	int deviceNumber;

	int* count = new int[3];
	count[0] = count[1] = count[2] = 1;

	int* offset = new int[3];
	offset[0] = offset[1] = offset[2] = 0;

	int total = 1;


	in.read((char*)&dimension, SIZE_INT);
	in.read((char*)&node, SIZE_INT);
	in.read((char*)&deviceType, SIZE_INT);
	in.read((char*)&deviceNumber, SIZE_INT);

	cout << endl;
	cout << "Block #" << idx << endl;
	cout << "	dimension:     " << dimension << endl;
	cout << "	node:          " << node << endl;
	cout << "	device type:   " << deviceType << endl;
	cout << "	device number: " << deviceNumber << endl;

	for (int j = 0; j < dimension; ++j) {
		in.read((char*)&offset[j], SIZE_INT);
		cout << "	offset" << j << ":           " << offset[j] << endl;
	}

	for (int j = 0; j < dimension; ++j) {
		in.read((char*)&count[j], SIZE_INT);
		cout << "	count" << j << ":            " << count[j] << endl;
		total *= count[j];
	}

	unsigned short int* initFuncNumber = new unsigned short int [total];
	unsigned short int* compFuncNumber = new unsigned short int [total];

	in.read((char*)initFuncNumber, total*SIZE_UN_SH_INT);
	in.read((char*)compFuncNumber, total*SIZE_UN_SH_INT);


	cout << "Init func number:" << endl;
	/*for (int idxY = 0; idxY < count[1]; ++idxY) {
		for (int idxX = 0; idxX < count[0]; ++idxX)
		    cout << initFuncNumber[idxY*count[0]+idxX] << " ";
		cout << endl;
	}
	cout << endl;*/

	cout << "Comp func number:" << endl;
	/*for (int idxY = 0; idxY < count[1]; ++idxY) {
		for (int idxX = 0; idxX < count[0]; ++idxX)
			cout << compFuncNumber[idxY*count[0]+idxX] << " ";
		cout << endl;
	}
	cout << endl;*/

	if(node == mWorldRank){
		if (deviceType==0)  //CPU BLOCK
			resBlock = new BlockCpu(idx, dimension, count[0], count[1], count[2], offset[0], offset[1], offset[2], node, deviceNumber, mHaloSize, mCellSize, initFuncNumber, compFuncNumber, mSolverIndex, mAtol, mRtol);
		else if (deviceType==1) //GPU BLOCK
			resBlock = new BlockGpu(idx, dimension, count[0], count[1], count[2], offset[0], offset[1], offset[2], node, deviceNumber, mHaloSize, mCellSize, initFuncNumber, compFuncNumber, mSolverIndex, mAtol, mRtol);
		else{
			printf("Invalid block type!\n");
			assert(false);
		}
	}
	else
		resBlock =  new BlockNull(idx, dimension, count[0], count[1], count[2], offset[0], offset[1], offset[2], node, deviceNumber, mHaloSize, mCellSize);

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

	in.read((char*)&dimension, SIZE_INT);
	cout << endl;
	cout << "Interconnect #<NONE>" << endl;

	for (int j = 0; j < dimension; ++j) {
		in.read((char*)&length[j], SIZE_INT);
		cout << "	length" << j << ":           " << length[j] << endl;
	}

	in.read((char*)&sourceBlock, SIZE_INT);
	in.read((char*)&destinationBlock, SIZE_INT);
	in.read((char*)&sourceSide, SIZE_INT);
	in.read((char*)&destinationSide, SIZE_INT);

	cout << "	source block:      " << sourceBlock << endl;
	cout << "	destination block: " << destinationBlock << endl;
	cout << "	source side:       " << sourceSide << endl;
	cout << "	destination side:  " << destinationSide << endl;

	for (int j = 0; j < dimension; ++j) {
		in.read((char*)&offsetSource[j], SIZE_INT);
		cout << "	offsetSource" << j << ":            " << offsetSource[j] << endl;
	}

	for (int j = 0; j < dimension; ++j) {
		in.read((char*)&offsetDestination[j], SIZE_INT);
		cout << "	offsetDestnation" << j << ":        " << offsetDestination[j] << endl;
	}

	double* sourceData = mBlocks[sourceBlock]->addNewBlockBorder(mBlocks[destinationBlock], getSide(sourceSide), offsetSource[0], offsetSource[1], length[0], length[1]);
	double* destinationData = mBlocks[destinationBlock]->addNewExternalBorder(mBlocks[sourceBlock], getSide(destinationSide),
			offsetDestination[0], offsetDestination[1], length[0], length[1], sourceData);

	int sourceNode = mBlocks[sourceBlock]->getNodeNumber();
	int destinationNode = mBlocks[destinationBlock]->getNodeNumber();

	int borderLength = length[0] * length[1] * mCellSize * mHaloSize;

	//cout << endl << "ERROR sorceData = destinationData = NULL!!!" << endl;

	return new Interconnect(sourceNode, destinationNode, borderLength, sourceData, destinationData);
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
		if( isCPU(mBlocks[i]->getBlockType()) )
			count++;

	return count;
}

/*
 * Количество блоков, имеющих тип "видеокарта".
 */
int Domain::getGpuBlockCount() {
	int count = 0;
	for (int i = 0; i < mBlockCount; ++i)
		if(isGPU(mBlocks[i]->getBlockType()))
			count++;

	return count;
}

/*
 * Количество реальных блоков на этом потоке.
 */
int Domain::realBlockCount() {
	int count = 0;
	for (int i = 0; i < mBlockCount; ++i)
		if( mBlocks[i]->isRealBlock() )
			count++;

	return count;
}

void Domain::saveState(char* inputFile) {
	//printf("\nsaveState %f %f %f\n", counterSaveTime, saveInterval, currentTime);

	char saveFile[100];

	int length = lastChar(inputFile, '/');

	strncpy(saveFile, inputFile, length);
	saveFile[ length ] = 0;

	//printf("\nbefor save %f %f %f %s %s\n", counterSaveTime, saveInterval, currentTime, saveFile, inputFile);

	sprintf(saveFile, "%s%s%f%s", saveFile, "/project-", currentTime, ".bin");

	//printf("\nbefor save %f %f %f %s %s\n", counterSaveTime, saveInterval, currentTime, saveFile, inputFile);

	saveStateToFile( saveFile );
}

void Domain::saveStateToFile(char* path) {
	double** resultAll = collectDataFromNode();

	if( mWorldRank == 0 ) {
		ofstream out;
		out.open(path, ios::binary);

		char save_file_code = SAVE_FILE_CODE;
		char version_major = VERSION_MAJOR;
		char version_minor = VERSION_MINOR;

		out.write((char*)&save_file_code, SIZE_CHAR);
		out.write((char*)&version_major, SIZE_CHAR);
		out.write((char*)&version_minor, SIZE_CHAR);

		out.write((char*)&currentTime, SIZE_DOUBLE);

		for (int i = 0; i < mBlockCount; ++i) {
			int count = mBlocks[i]->getGridElementCount();
			out.write((char*)resultAll[i], SIZE_DOUBLE * count);
		}
	}

	if( resultAll != NULL ) {
		for (int i = 0; i < mBlockCount; ++i)
			delete resultAll[i];
		delete resultAll;
	}
}

void Domain::loadStateFromFile(char* dataFile) {
	cout << endl << "LOAD STATE DOESN'T WORK" << endl;
	return;
	/*readFromFile(blockLocation);

	ifstream in;
	in.open(dataFile, ios::binary);

	int save_file_code;
	int version_major;
	int version_minor;

	int tmp_blockCount;

	//in >> save_file_code;
	in.read((char*)&save_file_code, SIZE_INT);

	if( save_file_code != SAVE_FILE_CODE ) {
		cout << endl << "Error save file. Save code." << endl;
		exit(0);
	}

	//in >> version_major;
	//in >> version_minor;
	in.read((char*)&version_major, SIZE_INT);
	in.read((char*)&version_minor, SIZE_INT);

	//in >> currentTime;
	//in >> tmp_blockCount;

	in.read((char*)&currentTime, SIZE_DOUBLE);
	in.read((char*)&tmp_blockCount, SIZE_INT);

	if( tmp_blockCount != blockCount ) {
		cout << endl << "Error save file. Block count." << endl;
		exit(0);
	}

	for (int i = 0; i < blockCount; ++i) {
		int length, width;

		//in >> length;
		//in >> width;
		in.read((char*)&length, SIZE_INT);
		in.read((char*)&width, SIZE_INT);

		cout << length << "l " << width << "w ";

		if( length != mBlocks[i]->getLength() || width != mBlocks[i]->getWidth() ) {
			cout << endl << "Error save file. Block size." << endl;
			exit(0);
		}

		double* data = new double [length * width];

		for (int j = 0; j < length * width; ++j)
			//in >> data[j];
			in.read((char*)&(data[j]), SIZE_DOUBLE);

		mBlocks[i]->loadData(data);
		delete data;
	}*/
}

void Domain::printStatisticsInfo(char* inputFile, char* outputFile, double calcTime, char* statisticsFile) {
	//cout << endl << "PRINT STATISTIC INFO DOESN'T WORK" << endl;

	int count = 0;
	for (int i = 0; i < mBlockCount; ++i) {
		count += mBlocks[i]->getGridElementCount();
	}



    printf("\n\nSteps accepted: %d\nSteps rejected: %d\n", mAcceptedStepCount, mRejectedStepCount);
    int stepCount = mRejectedStepCount + mAcceptedStepCount;
    printf("Time: %.2f\nElement count: %d\nPerformance: %.2f\n\n", calcTime, count, (double)(count) * stepCount / calcTime);

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
	double** resultAll = collectDataFromNode();

		if( mWorldRank == 0 ) {
			for (int i = 0; i < mBlockCount; ++i) {
				int count = mBlocks[i]->getGridElementCount();

				for (int j = 0; j < count; ++j) {
					if(isnan(resultAll[i][j]))
						return true;
				}

			}
		}

		if( resultAll != NULL ) {
			for (int i = 0; i < mBlockCount; ++i)
				delete resultAll[i];
			delete resultAll;
		}


	return false;
}

void Domain::checkOptions(int flags, double _stopTime, char* saveFile) {
	if( flags & TIME_EXECUTION )
		setStopTime(_stopTime);

	if( flags & LOAD_FILE )
		loadStateFromFile(saveFile);
}
