/*
 * realblock.cpp
 *
 *  Created on: 15 окт. 2015 г.
 *      Author: frolov
 */

#include "../blocks/realblock.h"

using namespace std;

RealBlock::RealBlock(int _nodeNumber, int _dimension, int _xCount, int _yCount, int _zCount, int _xOffset, int _yOffset,
		int _zOffset, int _cellSize, int _haloSize, int _blockNumber, ProcessingUnit* _pu,
		unsigned short int* _initFuncNumber, unsigned short int* _compFuncNumber, Problem* _problem,
		NumericalMethod* _numericalMethod) :
		Block(_nodeNumber, _dimension, _xCount, _yCount, _zCount, _xOffset, _yOffset, _zOffset, _cellSize, _haloSize) {

	//printf("\nbefore create block\n");
	pu = _pu;

	blockNumber = _blockNumber;

	sendBorderInfo = NULL;
	tempSendBorderInfo.clear();

	receiveBorderInfo = NULL;
	tempReceiveBorderInfo.clear();

	blockBorder = NULL;
	tempBlockBorder.clear();

	externalBorder = NULL;
	tempExternalBorder.clear();

	countSendSegmentBorder = countReceiveSegmentBorder = 0;

	int nodeCount = getGridNodeCount();
	int elementCount = getGridElementCount();

	mCompFuncNumber = pu->newUnsignedShortIntArray(nodeCount);
	mInitFuncNumber = pu->newUnsignedShortIntArray(nodeCount);

	pu->copyArray(_compFuncNumber, mCompFuncNumber, nodeCount);
	pu->copyArray(_initFuncNumber, mInitFuncNumber, nodeCount);

	// TODO зачем mParamCount?
	int mParamsCount = 0;
	getFuncArray(&mUserFuncs, blockNumber);
	getInitFuncArray(&mUserInitFuncs);
	initDefaultParams(&mParams, &mParamsCount);

	mProblem = _problem;
	mNumericalMethod = _numericalMethod;

	int commonTempStoragesCount = mNumericalMethod->getCommonTempStorageCount();
	mCommonTempStorages = pu->newDoublePointerArray(commonTempStoragesCount);

	for (int i = 0; i < commonTempStoragesCount; ++i) {
		mCommonTempStorages[i] = pu->newDoubleArray(elementCount);
	}

	int stateCount = mProblem->getStateCount();
	mStates = new State*[stateCount];
	for (int i = 0; i < stateCount; ++i) {
		mStates[i] = new State(pu, mNumericalMethod, mCommonTempStorages, elementCount);
	}

	//TODO: getSourceStorage(0) заменить 0, возможно, нужна специальная функция
	//double* state = mStates[mProblem->getCurrentStateNumber()]->getSourceStorage(0);
	double* state = mStates[mProblem->getCurrentStateNumber()]->getState();
	//printf("\n%p %d\n", state, nodeCount);
	pu->initState(state, mUserInitFuncs, mInitFuncNumber, blockNumber, 0.0);

	int sourceLength = 1 + mProblem->getDelayCount();
	mSource = pu->newDoublePointerArray(sourceLength);
	mSource[0] = NULL;
	for (int i = 1; i < sourceLength; ++i) {
		mSource[i] = pu->newDoubleArray(elementCount);
	}

	mResult = NULL;
}

RealBlock::~RealBlock() {
	int sourceLength = 1 + mProblem->getDelayCount();
	for (int i = 1; i < sourceLength; ++i) {
		pu->deleteDeviceSpecificArray(mSource[i]);
	}

	pu->deleteDeviceSpecificArray(mSource);

	int stateCount = mProblem->getStateCount();
	for (int i = 0; i < stateCount; ++i) {
		delete mStates[i];
	}

	delete mStates;
}

/*void RealBlock::afterCreate(int problemType, int solverType, double aTol, double rTol) {
 problem = createProblem(problemType, solverType, aTol, rTol);

 double* state = problem->getCurrentStatePointer();
 pu->initState(state, mUserInitFuncs, mInitFuncNumber, blockNumber, 0.0);
 }*/

double* RealBlock::getNewBlockBorder(Block* neighbor, int borderLength) {
	if ((nodeNumber == neighbor->getNodeNumber()) && neighbor->isProcessingUnitGPU()) {
		return pu->newDoublePinnedArray(borderLength);
	} else {
		return pu->newDoubleArray(borderLength);
	}
}

double* RealBlock::getNewExternalBorder(Block* neighbor, int borderLength, double* border) {
	if (nodeNumber == neighbor->getNodeNumber()) {
		return border;
	} else {
		return pu->newDoubleArray(borderLength);
	}
}

/*ProblemType* RealBlock::createProblem(int problemType, int solverType, double aTol, double rTol) {
 int elementCount = getGridElementCount();

 switch (problemType) {
 case ORDINARY:
 return new Ordinary(pu, solverType, elementCount, aTol, rTol);

 case DELAY:
 printf("\nDELAY PROBLEM TYPE NOT READY!!!\n");
 return NULL;

 default:
 return new Ordinary(pu, solverType, elementCount, aTol, rTol);
 }
 }*/

void RealBlock::computeStageBorder(int stage, double time) {
	pu->computeBorder(mUserFuncs, mCompFuncNumber, mResult, mSource, time, mParams, externalBorder, zCount, yCount,
			xCount, haloSize);
}

void RealBlock::computeStageCenter(int stage, double time) {
	/*double* result = problem->getResult(stage);
	 double** source = problem->getSource(stage);*/
	//printf("\n\n###\n\n");
	//printf("\nresult %p\nsource 0 %p\n", mResult, mSource[0]);
	pu->computeCenter(mUserFuncs, mCompFuncNumber, mResult, mSource, time, mParams, externalBorder, zCount, yCount,
			xCount, haloSize);
}

void RealBlock::prepareArgument(int stage, double timeStep) {
	//problem->prepareArgument(stage, timestep);
	int currentStateNumber = mProblem->getCurrentStateNumber();
	mStates[currentStateNumber]->prepareArgument(timeStep, stage);
}

void RealBlock::prepareStageData(int stage) {
	//double* source = problem->getCurrentStateStageData(stage);
	// TODO: ПРОВЕРИТЬ!!!
	double* source = mSource[0];
	for (int i = 0; i < countSendSegmentBorder; ++i) {
		double* result = blockBorder[i];

		int index = INTERCONNECT_COMPONENT_COUNT * i;

		int mStart = sendBorderInfo[index + M_OFFSET];
		int mStop = mStart + sendBorderInfo[index + M_LENGTH];

		int nStart = sendBorderInfo[index + N_OFFSET];
		int nStop = nStart + sendBorderInfo[index + N_LENGTH];

		switch (sendBorderInfo[index + SIDE]) {
			case LEFT:
				pu->prepareBorder(result, source, mStart, mStop, nStart, nStop, 0 + 1, haloSize + 1, yCount, xCount, cellSize);
				break;
			case RIGHT:
				pu->prepareBorder(result, source, mStart, mStop, nStart, nStop, xCount - haloSize - 1, xCount - 1, yCount,
						xCount, cellSize);
				break;
			case FRONT:
				pu->prepareBorder(result, source, mStart, mStop, 0 + 1, haloSize + 1, nStart, nStop, yCount, xCount, cellSize);
				break;
			case BACK:
				pu->prepareBorder(result, source, mStart, mStop, yCount - haloSize - 1, yCount - 1, nStart, nStop, yCount,
						xCount, cellSize);
				break;
			case TOP:
				pu->prepareBorder(result, source, 0 + 1, haloSize + 1, mStart, mStop, nStart, nStop, yCount, xCount, cellSize);
				break;
			case BOTTOM:
				pu->prepareBorder(result, source, zCount - haloSize - 1, zCount - 1, mStart, mStop, nStart, nStop, yCount,
						xCount, cellSize);
				break;
			default:
				break;
		}
	}
}

void RealBlock::prepareStageSourceResult(int stage, double timeStep, double currentTime) {
	int currentStateNumber = mProblem->getCurrentStateNumber();
	int delayCount = mProblem->getDelayCount();

	mResult = mStates[currentStateNumber]->getResultStorage(stage);/*problem->getResult(stage);*/
	/*TODO: Возможные проблемы при работе с видеокартой.
	 Нельзя вносить изменение в mSource[i] через CPU, необходима специальная функция в ProcessingUnit*/
	mSource[0] = mStates[currentStateNumber]->getSourceStorage(stage);/*problem->getSource(stage);*/

	//TODO: Унификация цикла с конструктором класса. sourseLength или иной вариант
	for (int i = 0; i < delayCount; ++i) {
		/*TODO: Реализовать в классе Problem метод, который выясняет требуется ли "плотный" вывод или
		 * необходимо воспользоваться готовыми функциями от пользователя, которые дают конкретные значения в прошлом*/
		if (mProblem->getDelay(i) < currentTime) {
			int delayStateNumber = mProblem->getStateNumberForDelay(i);
			/* TODO: Расчет "плотного" вывода должен осуществляться с учетом стадии
			 * но сами расчеты ведуться от mState
			 */
			// TODO: Вычислять в проблеме. Доставать из проблемы
			double theta = mProblem->getTethaForDelay(i);
			mStates[delayStateNumber]->computeDenseOutput(timeStep, theta, mSource[1 + i]);
			//pu->printArray(mSource[1 + i], 1, 1, 11, 1);
		} else {
			// TODO: Создать ПРАВИЛЬНЫЕ функции для работы с состояниями в прошлом
			pu->delayFunction(mSource[1 + i], mUserInitFuncs, mInitFuncNumber, blockNumber,
					currentTime - mProblem->getDelay(i));
		}

		/*pu->printArray(mSource[0], 1, 1, 11, 1);
		pu->printArray(mSource[1 + i], 1, 1, 11, 1);*/
	}
}

bool RealBlock::isRealBlock() {
	return true;
}

int RealBlock::getBlockType() {
	return pu->getType();
}

int RealBlock::getDeviceNumber() {
	return pu->getDeviceNumber();
}

bool RealBlock::isBlockType(int type) {
	return pu->isDeviceType(type);
}

bool RealBlock::isDeviceNumber(int number) {
	return pu->isDeviceNumber(number);
}

bool RealBlock::isProcessingUnitCPU() {
	return pu->isCPU();
}

bool RealBlock::isProcessingUnitGPU() {
	return pu->isGPU();
}

double RealBlock::getStepError(double timeStep) {
	//return problem->getStepError(timestep);
	int currentStateNumber = mProblem->getCurrentStateNumber();
	return mStates[currentStateNumber]->computeStepError(timeStep);
}

void RealBlock::confirmStep(double timeStep) {
	//problem->confirmStep(timestep);
	int currentStateNumber = mProblem->getCurrentStateNumber();
	int nextStateNumber = mProblem->getNextStateNumber();

	mStates[currentStateNumber]->confirmStep(timeStep, mStates[nextStateNumber], (ISmartCopy*) mProblem);
}

void RealBlock::rejectStep(double timeStep) {
	//problem->rejectStep(timestep);
	int currentStateNumber = mProblem->getCurrentStateNumber();
	mStates[currentStateNumber]->rejectStep(timeStep);
}

double* RealBlock::addNewBlockBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength) {
	countSendSegmentBorder++;

	tempSendBorderInfo.push_back(side);
	tempSendBorderInfo.push_back(mOffset);
	tempSendBorderInfo.push_back(nOffset);
	tempSendBorderInfo.push_back(mLength);
	tempSendBorderInfo.push_back(nLength);

	int borderLength = mLength * nLength * cellSize * haloSize;

	double* newBlockBorder = getNewBlockBorder(neighbor, borderLength);

	tempBlockBorder.push_back(newBlockBorder);

	return newBlockBorder;
}

double* RealBlock::addNewExternalBorder(Block* neighbor, int side, int mOffset, int nOffset, int mLength, int nLength,
		double* border) {
	countReceiveSegmentBorder++;

	tempReceiveBorderInfo.push_back(side);
	tempReceiveBorderInfo.push_back(mOffset);
	tempReceiveBorderInfo.push_back(nOffset);
	tempReceiveBorderInfo.push_back(mLength);
	tempReceiveBorderInfo.push_back(nLength);

	int borderLength = mLength * nLength * cellSize * haloSize;

	double* newExternalBorder = getNewExternalBorder(neighbor, borderLength, border);

	tempExternalBorder.push_back(newExternalBorder);

	//printf("\nExternal %d %d %d\n", nodeNumber, blockNumber, newExternalBorder);

	return newExternalBorder;
}

void RealBlock::moveTempBorderVectorToBorderArray() {
	blockBorder = pu->newDoublePointerArray(countSendSegmentBorder);

	sendBorderInfo = pu->newIntArray(INTERCONNECT_COMPONENT_COUNT * countSendSegmentBorder);

	externalBorder = pu->newDoublePointerArray(countReceiveSegmentBorder);

	receiveBorderInfo = pu->newIntArray(INTERCONNECT_COMPONENT_COUNT * countReceiveSegmentBorder);

	for (int i = 0; i < countSendSegmentBorder; ++i) {
		blockBorder[i] = tempBlockBorder.at(i);

		int index = INTERCONNECT_COMPONENT_COUNT * i;
		sendBorderInfo[index + SIDE] = tempSendBorderInfo.at(index + 0);
		sendBorderInfo[index + M_OFFSET] = tempSendBorderInfo.at(index + 1);
		sendBorderInfo[index + N_OFFSET] = tempSendBorderInfo.at(index + 2);
		sendBorderInfo[index + M_LENGTH] = tempSendBorderInfo.at(index + 3);
		sendBorderInfo[index + N_LENGTH] = tempSendBorderInfo.at(index + 4);
	}

	for (int i = 0; i < countReceiveSegmentBorder; ++i) {
		externalBorder[i] = tempExternalBorder.at(i);

		int index = INTERCONNECT_COMPONENT_COUNT * i;
		receiveBorderInfo[index + SIDE] = tempReceiveBorderInfo.at(index + 0);
		receiveBorderInfo[index + M_OFFSET] = tempReceiveBorderInfo.at(index + 1);
		receiveBorderInfo[index + N_OFFSET] = tempReceiveBorderInfo.at(index + 2);
		receiveBorderInfo[index + M_LENGTH] = tempReceiveBorderInfo.at(index + 3);
		receiveBorderInfo[index + N_LENGTH] = tempReceiveBorderInfo.at(index + 4);
	}

	tempBlockBorder.clear();
	tempExternalBorder.clear();

	tempSendBorderInfo.clear();
	tempReceiveBorderInfo.clear();
}

void RealBlock::getCurrentState(double* result) {
	//mProblem->getCurrentState(result);
}

void RealBlock::saveStateForDraw(char* path) {
	//mProblem->saveStateForDraw(path);
	/*int currentStateNumber = mProblem->getCurrentStateNumber();
	 mStates[currentStateNumber]->saveGeneralStorage(path);*/
	mProblem->savaDataForDraw(path, mStates);
}

void RealBlock::saveStateForLoad(char* path) {
	mProblem->saveData(path, mStates);
}

void RealBlock::saveStateForDrawDenseOutput(char* path, double timeStep, double tetha) {
	mProblem->saveStateForDrawDenseOutput(path, mStates, timeStep, tetha);
}

void RealBlock::loadState(ifstream& in) {
	//mProblem->loadState(in);
	mProblem->loadData(in, mStates);
}

bool RealBlock::isNan() {
	/*if (problem->isNan()) {
	 printf("\nBlock #%d, Node number = %d: NAN VALUE!\n", blockNumber, nodeNumber);
	 return true;
	 }
	 return false;*/
	int currentStateNumber = mProblem->getCurrentStateNumber();
	if (mStates[currentStateNumber]->isNan()) {
		printf("\nBlock #%d, Node number = %d: NAN VALUE!\n", blockNumber, nodeNumber);
		return true;
	}
	return false;
}

void RealBlock::print() {
	printf("################################################################################");
	printGeneralInformation();
	printBorderInfo();
	printData();
	printf("################################################################################");
	printf("\n\n\n");
}

void RealBlock::printGeneralInformation() {
	printf("\nBlock #%d\n"
			"   Node number: %d\n"
			"   xCount:      %d\n"
			"   yCount:      %d\n"
			"   zCount:      %d\n"
			"   xOffset:     %d\n"
			"   yOffset:     %d\n"
			"   zOffset:     %d\n"
			"   Cell size:   %d\n"
			"   Halo size:   %d\n"
			"\n", blockNumber, nodeNumber, xCount, yCount, zCount, xOffset, yOffset, zOffset, cellSize, haloSize);
}

void RealBlock::printBorderInfo() {
	int index = 0;

	printf("Send border info\n");
	for (int i = 0; i < countSendSegmentBorder; ++i) {
		index = INTERCONNECT_COMPONENT_COUNT * i;
		printf("Border #%d\n"
				"   Side    : %s\n"
				"   M_Offset: %d\n"
				"   N_Offset: %d\n"
				"   M_Length: %d\n"
				"   N_Length: %d\n"
				"   Address : %p\n"
				"\n", i, Utils::getSideName(sendBorderInfo[index + SIDE]), sendBorderInfo[index + M_OFFSET],
				sendBorderInfo[index + N_OFFSET], sendBorderInfo[index + M_LENGTH], sendBorderInfo[index + N_LENGTH],
				blockBorder[i]);
	}

	printf("\n");

	printf("Receive border info\n");
	for (int i = 0; i < countReceiveSegmentBorder; ++i) {
		index = INTERCONNECT_COMPONENT_COUNT * i;
		printf("Border #%d\n"
				"   Side    : %s\n"
				"   M_Offset: %d\n"
				"   N_Offset: %d\n"
				"   M_Length: %d\n"
				"   N_Length: %d\n"
				"   Address : %p\n"
				"\n", i, Utils::getSideName(receiveBorderInfo[index + SIDE]), receiveBorderInfo[index + M_OFFSET],
				receiveBorderInfo[index + N_OFFSET], receiveBorderInfo[index + M_LENGTH],
				receiveBorderInfo[index + N_LENGTH], externalBorder[i]);
	}

	printf("\n");
}

void RealBlock::printData() {
	//mProblem->print(zCount, yCount, xCount, cellSize);
	int currentStateNumber = mProblem->getCurrentStateNumber();
	mStates[currentStateNumber]->print(zCount, yCount, xCount, cellSize);
}
