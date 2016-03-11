/*
 * Domain.cpp
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#include "domain.h"
#include <cassert>

using namespace std;

Domain::Domain(int _world_rank, int _world_size, char* inputFile) {
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
	}

	//mJobId = _jobId;

	currentTime = 0;
	mStepCount = 0;

	timeStep = 0;
	stopTime = 0;

	mRepeatCount = 0;

	counterSaveTime = 0;

	dimension = 0;

	cpu = NULL;
	gpu0 = NULL;
	gpu1 = NULL;
	gpu2 = NULL;

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
	/*if(gpu0)
	 delete gpu0;
	 if(gpu1)
	 delete gpu1;
	 if(gpu2)
	 delete gpu2;*/
}

void Domain::compute(char* inputFile) {
	cout << endl << "Computation started..." << mWorkerRank << endl;
	cout << "Current time: " << currentTime << ", finish time: " << stopTime << ", time step: " << timeStep << endl;
	cout << "solver stage count: " << mSolverInfo->getStageCount() << endl;

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
	int userStatus = US_RUN;
	int jobState = JS_RUNNING;
	MPI_Bcast(&userStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (mPythonMaster && (mWorkerRank == 0))
		MPI_Send(&jobState, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

	cout << "Initial user status received: " << userStatus << endl;

	while ((userStatus != US_STOP) && (jobState == JS_RUNNING)) {
		nextStep();
		if (mPythonMaster && (mWorkerRank == 0)) {
			MPI_Send(&mLastStepAccepted, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
			MPI_Send(&timeStep, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
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
			percentage = newPercentage;
			if (mPythonMaster && (mWorkerRank == 0))
				MPI_Send(&percentage, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		}

		counterSaveTime += timeStep;

		int readyToSave = (saveInterval != 0) && (counterSaveTime > saveInterval);
		if (mPythonMaster && (mWorkerRank == 0))
			MPI_Send(&readyToSave, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
		if (!(currentTime < stopTime))
			jobState = JS_FINISHED;
		if (mPythonMaster && (mWorkerRank == 0))
			MPI_Send(&jobState, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);

		if (readyToSave) {
			counterSaveTime = 0;
			saveState(inputFile);
		}

		//check for termination request
		MPI_Bcast(&userStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);

	}
	cout << "Computation finished for worker #" << mWorkerRank << endl;
	//if ((mWorkerRank == 0)&&(!mPythonMaster))
	//    setDbJobState(JS_FINISHED);

}

void Domain::initSolvers() {
	computeStage(SOLVER_INIT_STAGE);
}

void Domain::computeStage(int stage) {
	//printBlocksToConsole();

	prepareData(stage);

	for (int i = 0; i < mConnectionCount; ++i)
		mInterconnects[i]->transfer();

	computeOneStepCenter(stage);

	for (int i = 0; i < mConnectionCount; ++i)
		mInterconnects[i]->wait();

	computeOneStepBorder(stage);

	prepareNextStageArgument(stage);
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
		//!!! только 0, рассылать
		timeStep = mSolverInfo->getNewStep(timeStep, error, totalGridElementCount);

		//!!! только 0, рассылать
		if (mSolverInfo->isErrorPermissible(error, totalGridElementCount)) {
			confirmStep(); //uses new timestep
			mAcceptedStepCount++;
			currentTime += timeStep;
			//cout<<"Step accepted\n"<<endl;
		} else {
			rejectStep(); //uses new timestep
			mRejectedStepCount++;
			cout << "Step rejected!\n" << endl;
		}

		printf("new time step = %f\n", timeStep);
	} else { //constant step
		confirmStep();
		mAcceptedStepCount++;
		currentTime += timeStep;

	}
}

void Domain::prepareDeviceData(int deviceType, int deviceNumber, int stage) {
	for (int i = 0; i < mBlockCount; ++i)
		if (mBlocks[i]->getBlockType() == deviceType && mBlocks[i]->getDeviceNumber() == deviceNumber) {
			//printf("\nSuccses\n");
			mBlocks[i]->prepareStageData(stage);
		}
}

void Domain::processDeviceBlocksBorder(int deviceType, int deviceNumber, int stage) {
	for (int i = 0; i < mBlockCount; ++i)
		if (mBlocks[i]->getBlockType() == deviceType && mBlocks[i]->getDeviceNumber() == deviceNumber) {
			//cout << endl << "ERROR! PROCESS DEVICE!" << endl;
			mBlocks[i]->computeStageBorder(stage, currentTime);
		}
}

void Domain::processDeviceBlocksCenter(int deviceType, int deviceNumber, int stage) {
	for (int i = 0; i < mBlockCount; ++i)
		if (mBlocks[i]->getBlockType() == deviceType && mBlocks[i]->getDeviceNumber() == deviceNumber) {
			//cout << endl << "ERROR! PROCESS DEVICE!" << endl;
			mBlocks[i]->computeStageCenter(stage, currentTime);
		}
}
void Domain::prepareDeviceArgument(int deviceType, int deviceNumber, int stage) {
	for (int i = 0; i < mBlockCount; ++i)
		if (mBlocks[i]->getBlockType() == deviceType && mBlocks[i]->getDeviceNumber() == deviceNumber) {
			//cout << endl << "ERROR! PROCESS DEVICE!" << endl;
			mBlocks[i]->prepareArgument(stage, timeStep);
		}
}

double Domain::getDeviceError(int deviceType, int deviceNumber) {
	double error = 0;
	for (int i = 0; i < mBlockCount; ++i)
		if (mBlocks[i]->getBlockType() == deviceType && mBlocks[i]->getDeviceNumber() == deviceNumber) {
			//cout << endl << "ERROR! PROCESS DEVICE!" << endl;
			error += mBlocks[i]->getStepError(timeStep);
		}
	return error;
}

void Domain::prepareData(int stage) {
#pragma omp task
	prepareDeviceData(GPU_UNIT, 0, stage);
#pragma omp task
	prepareDeviceData(GPU_UNIT, 1, stage);
#pragma omp task
	prepareDeviceData(GPU_UNIT, 2, stage);

	prepareDeviceData(CPU_UNIT, 0, stage);

#pragma omp taskwait
}

void Domain::computeOneStepBorder(int stage) {
#pragma omp task
	processDeviceBlocksBorder(GPU_UNIT, 0, stage);
#pragma omp task
	processDeviceBlocksBorder(GPU_UNIT, 1, stage);
#pragma omp task
	processDeviceBlocksBorder(GPU_UNIT, 2, stage);

	processDeviceBlocksBorder(CPU_UNIT, 0, stage);
}

void Domain::prepareNextStageArgument(int stage) {
#pragma omp task
	prepareDeviceArgument(GPU_UNIT, 0, stage);
#pragma omp task
	prepareDeviceArgument(GPU_UNIT, 1, stage);
#pragma omp task
	prepareDeviceArgument(GPU_UNIT, 2, stage);

	prepareDeviceArgument(CPU_UNIT, 0, stage);
}

void Domain::computeOneStepCenter(int stage) {
#pragma omp task
	processDeviceBlocksCenter(GPU_UNIT, 0, stage);
#pragma omp task
	processDeviceBlocksCenter(GPU_UNIT, 1, stage);
#pragma omp task
	processDeviceBlocksCenter(GPU_UNIT, 2, stage);

	processDeviceBlocksCenter(CPU_UNIT, 0, stage);
}

//TODO next two methods are not parallel!
void Domain::confirmStep() {
	mLastStepAccepted = 1;
	for (int i = 0; i < mBlockCount; ++i) {
		mBlocks[i]->confirmStep(timeStep);
	}
}

void Domain::rejectStep() {
	mLastStepAccepted = 0;
	for (int i = 0; i < mBlockCount; ++i) {
		mBlocks[i]->rejectStep(timeStep);
	}
}

double Domain::collectError() {
	double err1, err2, err3;
	err1 = err2 = err3 = 0;
	//1. Get total error for current node
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
	return totalError;
}

void Domain::printBlocksToConsole() {
	for (int i = 0; i < mBlockCount; ++i) {
		if (mBlocks[i]->isRealBlock())
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

	in.read((char*) &dimension, SIZE_INT);
	createProcessigUnit();

	readCellAndHaloSize(in);
	readSolverIndex(in);
	readSolverTolerance(in);

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

	readBlockCount(in);

	mBlocks = new Block*[mBlockCount];

	for (int i = 0; i < mBlockCount; ++i)
		mBlocks[i] = readBlock(in, i, dimension);

	readConnectionCount(in);

	mInterconnects = new Interconnect*[mConnectionCount];

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
	in.read((char*) &timeStep, SIZE_DOUBLE);

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
		else if (deviceType == 1) //GPU BLOCK
			switch (deviceNumber) {
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
			}
		else {
			printf("Invalid block type!\n");
			assert(false);
		}

		printf("\nPROBLEM TYPE ALWAYS = ORDINARY!!!\n");

		resBlock = new RealBlock(node, dimension, count[0], count[1], count[2], offset[0], offset[1], offset[2],
				mCellSize, mHaloSize, idx, pu, initFuncNumber, compFuncNumber, ORDINARY, mSolverIndex, mAtol, mRtol);
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

	double* sourceData = mBlocks[sourceBlock]->addNewBlockBorder(mBlocks[destinationBlock], getSide(sourceSide),
			offsetSource[0], offsetSource[1], length[0], length[1]);
	double* destinationData = mBlocks[destinationBlock]->addNewExternalBorder(mBlocks[sourceBlock],
			getSide(destinationSide), offsetDestination[0], offsetDestination[1], length[0], length[1], sourceData);

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
		if (isCPU(mBlocks[i]->getBlockType()))
			count++;

	return count;
}

/*
 * Количество блоков, имеющих тип "видеокарта".
 */
int Domain::getGpuBlockCount() {
	int count = 0;
	for (int i = 0; i < mBlockCount; ++i)
		if (isGPU(mBlocks[i]->getBlockType()))
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

void Domain::saveState(char* inputFile) {
	//printf("\nsaveState %f %f %f\n", counterSaveTime, saveInterval, currentTime);

	char saveFile[250];

	/*int length = Utils::lastChar(inputFile, '/');

	 strncpy(saveFile, inputFile, length);

	 saveFile[length] = 0;*/
	Utils::copyToLastChar(saveFile, inputFile, '/');

	sprintf(saveFile, "%s%s%f%s", saveFile, "project-", currentTime, ".bin");

	saveStateToFile(saveFile);
}

void Domain::saveStateToFile(char* path) {
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
		out.write((char*) &timeStep, SIZE_DOUBLE);

		out.close();
	}

	for (int i = 0; i < mBlockCount; ++i) {
		mBlocks[i]->saveState(path);
		MPI_Barrier(mWorkerComm);
	}
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
	in.read((char*) &timeStep, SIZE_DOUBLE);

	currentTime = fileCurrentTime;

	for (int i = 0; i < mBlockCount; ++i) {
		mBlocks[i]->loadState(in);
	}

	in.close();
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

		FILE* out;
		out = fopen(statFile, "a");

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

int Domain::getMaximumNumberSavedStates() {
	return 0;
}

void Domain::checkOptions(int flags, double _stopTime, char* saveFile) {
	if (flags & TIME_EXECUTION)
		setStopTime(_stopTime);

	if (flags & LOAD_FILE)
		loadStateFromFile(saveFile);
}

void Domain::createProcessigUnit() {
	switch (dimension) {
		case 1:
			cpu = new CPU_1d(0);
			break;
		case 2:
			cpu = new CPU_2d(0);
			break;
		case 3:
			cpu = new CPU_3d(0);
			break;
		default:
			break;
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

/*
 void Domain::storeDbFileName(char* inputFile){
 char saveFile[100];

 int length = lastChar(inputFile, '/');

 strncpy(saveFile, inputFile, length);
 saveFile[ length ] = 0;

 sprintf(saveFile, "%s%s%f%s", saveFile, "/project-", currentTime, ".bin");


 dbConnStoreFileName(mJobId, saveFile);
 }*/

