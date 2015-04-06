/*
 * Domain.cpp
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#include "domain.h"

using namespace std;

Domain::Domain(int _world_rank, int _world_size, char* inputFile, int _flags, int _stepCount, double _stopTime, char* loadFile) {
	world_rank = _world_rank;
	world_size = _world_size;

	currentTime = 0;
	stepCount = 0;

	stepTime = 0;
	stopTime = 0;

	repeatCount = 0;


	flags = _flags;

	if( flags & LOAD_FILE )
		loadStateFromFile(inputFile, loadFile);
	else
		readFromFile(inputFile);

	if( flags & TIME_EXECUTION )
		stopTime = _stopTime;

	if( flags & STEP_EXECUTION )
		stepCount = _stepCount;
}

Domain::~Domain() {
	for (int i = 0; i < blockCount; ++i)
		delete mBlocks[i];
	delete mBlocks;

	for (int i = 0; i < connectionCount; ++i)
		delete mInterconnects[i];
	delete mInterconnects;
}

double** Domain::collectDataFromNode() {
	double** resultAll = NULL;

	if(world_rank == 0) {
		resultAll = new double* [blockCount];
	}

	for (int i = 0; i < blockCount; ++i) {
		double* tmp = getBlockCurrentState(i);

		if(world_rank == 0)
			resultAll[i] = tmp;
	}

	return resultAll;
}

double* Domain::getBlockCurrentState(int number) {
	cout << endl << "GET CURRENT STATE NOT WORK!" << endl;
	return 0;
	/*double* result = NULL;

	if(world_rank == 0) {
		if(mBlocks[number]->isRealBlock()) {
			result = mBlocks[number]->getCurrentState();
		}
		else {
			result = new double [mBlocks[number]->getLength() * mBlocks[number]->getWidth()];
			MPI_Recv(result, mBlocks[number]->getLength() * mBlocks[number]->getWidth(), MPI_DOUBLE, mBlocks[number]->getNodeNumber(), 999, MPI_COMM_WORLD, &status);
		}

		return result;
	}
	else {
		if(mBlocks[number]->isRealBlock()) {
			result = mBlocks[number]->getCurrentState();
			MPI_Send(result, mBlocks[number]->getLength() * mBlocks[number]->getWidth(), MPI_DOUBLE, 0, 999, MPI_COMM_WORLD);
			delete result;
			return NULL;
		}
	}
	return NULL;*/
}

void Domain::compute(char* saveFile) {
	/*
	 * Вычисление коэффициентов необходимых для расчета теплопроводности

	double dX = 1./widthArea;
	double dY = 1./lengthArea;


	 * Аналогично вышенаписанному

	double dX2 = dX * dX;
	double dY2 = dY * dY;

	double dT = ( dX2 * dY2 ) / ( 2 * ( dX2 + dY2 ) );


	 * Выполнение

	if( flags & STEP_EXECUTION)
		for (int i = 0; i < stepCount; i++) {
			nextStep(dX2, dY2, dT);
			currentTime += stepTime;
			repeatCount++;
		}
	else
		while ( currentTime < stopTime ) {
			nextStep(dX2, dY2, dT);
			currentTime += stepTime;
			repeatCount++;
		}

	if( flags & SAVE_FILE )
		saveStateToFile(saveFile);*/
}

void Domain::nextStep(double dX2, double dY2, double dT) {
	prepareData();

	for (int i = 0; i < connectionCount; ++i)
		mInterconnects[i]->sendRecv(world_rank);

	computeOneStepCenter(dX2, dY2, dT);

	for (int i = 0; i < connectionCount; ++i)
		mInterconnects[i]->wait();

	computeOneStepBorder(dX2, dY2, dT);

	swapBlockMatrix();
}

void Domain::prepareDeviceData(int deviceType, int deviceNumber) {
	for (int i = 0; i < blockCount; ++i)
		if( mBlocks[i]->getBlockType() == deviceType && mBlocks[i]->getDeviceNumber() == deviceNumber ) {
			mBlocks[i]->prepareData();
		}
}

void Domain::processDeviceBlocksBorder(int deviceType, int deviceNumber, double dX2, double dY2, double dT) {
	for (int i = 0; i < blockCount; ++i)
        if( mBlocks[i]->getBlockType() == deviceType && mBlocks[i]->getDeviceNumber() == deviceNumber ) {
        	cout << endl << "ERROR! PROCESS DEVICE!" << endl;
		    mBlocks[i]->computeOneStepBorder(currentTime, NULL);
		}
}

void Domain::processDeviceBlocksCenter(int deviceType, int deviceNumber, double dX2, double dY2, double dT) {
	for (int i = 0; i < blockCount; ++i)
        if( mBlocks[i]->getBlockType() == deviceType && mBlocks[i]->getDeviceNumber() == deviceNumber ) {
        	cout << endl << "ERROR! PROCESS DEVICE!" << endl;
		    mBlocks[i]->computeOneStepCenter(currentTime, NULL);
		}
}

void Domain::prepareData() {
#pragma omp task
	prepareDeviceData(GPU, 0);
#pragma omp task
	prepareDeviceData(GPU, 1);
#pragma omp task
	prepareDeviceData(GPU, 2);

	prepareDeviceData(CPU, 0);

#pragma omp taskwait
}

void Domain::computeOneStepBorder(double dX2, double dY2, double dT) {
#pragma omp task
	processDeviceBlocksBorder(GPU, 0, dX2, dY2, dT);
#pragma omp task
	processDeviceBlocksBorder(GPU, 1, dX2, dY2, dT);
#pragma omp task
	processDeviceBlocksBorder(GPU, 2, dX2, dY2, dT);

	processDeviceBlocksBorder(CPU, 0, dX2, dY2, dT);
}

void Domain::computeOneStepCenter(double dX2, double dY2, double dT) {
#pragma omp task
	processDeviceBlocksCenter(GPU, 0, dX2, dY2, dT);
#pragma omp task
	processDeviceBlocksCenter(GPU, 1, dX2, dY2, dT);
#pragma omp task
	processDeviceBlocksCenter(GPU, 2, dX2, dY2, dT);

	processDeviceBlocksCenter(CPU, 0, dX2, dY2, dT);
}

void Domain::swapBlockMatrix() {
	for (int i = 0; i < blockCount; ++i) {
		mBlocks[i]->swapMatrix();
	}
}

void Domain::print(char* path) {
	cout << endl << "PRINT DON'T WORK" << endl;
	return;
	/*double** resultAll = collectDataFromNode();
	double** area = NULL;

	if( world_rank == 0 ) {
		area = new double* [lengthArea];
		for (int i = 0; i < lengthArea; ++i) {
			area[i] = new double [widthArea];
		}

		for (int i = 0; i < lengthArea; ++i) {
			for (int j = 0; j < widthArea; ++j) {
				area[i][j] = 0;
			}
		}

		for (int i = 0; i < blockCount; ++i) {
			int length_move = mBlocks[i]->getLengthMove();
			int width_move = mBlocks[i]->getWidthMove();

			int length = mBlocks[i]->getLength();
			int width = mBlocks[i]->getWidth();

			for (int j = 0; j < length; ++j) {
				for (int k = 0; k < width; ++k) {
					area[ j + length_move ][ k + width_move ] =
							resultAll[i][ j * width + k ];
				}
			}
		}

		ofstream out;
		out.open(path);

		for (int i = 0; i < lengthArea; ++i) {
			for (int j = 0; j < widthArea; ++j)
				out << i << " " << j << " " << area[i][j] << endl;
			out << endl;
		}

		out.close();
	}

	if( resultAll != NULL ) {
		for (int i = 0; i < blockCount; ++i)
			delete resultAll[i];
		delete resultAll;
	}

	if( area != NULL ) {
		for (int i = 0; i < blockCount; ++i)
			delete area[i];
		delete area;
	}*/
}

void Domain::printAreaToConsole() {
	cout << endl << "PRINT AREA TO CONSOLE DON'T WORK" << endl;
	return;
	/*double** resultAll = collectDataFromNode();

	cout.setf(ios::fixed);
	for(int i = 0; i < lengthArea; i++) {
		for( int j = 0; j < widthArea; j++ ) {
			cout.width(7);
			cout.precision(1);
			cout << resultAll[i][j];
		}
		cout << endl;
	}

	if( resultAll != NULL ) {
		for (int i = 0; i < lengthArea; ++i)
			delete resultAll[i];
		delete resultAll;
	}*/
}

void Domain::printBlocksToConsole() {
	for (int i = 0; i < blockCount; ++i) {
		mBlocks[i]->print();
	}
}

/*
 * Функция чтения данных из файла
 *
 * 50 40 - размеры области
 * 2 - количество блоков, которые находятся в этой области
 *
 * 40 20 - размеры блока
 * 0 0 - координаты блока в области, смещения по горизонтали и вертикали.
 * 0 - номер потока, который ДОЛЖЕН РЕАЛЬНО создать этот блок.
 * 0 - тип блока, 0 - центальный процессор, 1 - первая видеокарта узла, 2 - вторая, 3 - третья.
 *
 * 40 20
 * 10 20
 * 1
 * 1
 *
 * 2 - число связей для данной задачи
 * 0 1 l 10 0 30 - блок с номером "0" передает блоку с номером "1" данные. l - левая граница блока "1" (t - верхняя, b - нижняя, r - правая), указывается сторона блока-получателя.
 * 					сдвиг по границе для блока-источника - 10, для блока-получателя - 0.
 * 1 0 r 0 10 30 - обратная связь для этих блоков. Этой связи может не быть, тогда граница будет проницаема только в одну сторону.
 *
 * Так выглядит описанная выше область
 *
 * # # # # # # # # # # # # # # # # # # # #
 * # # # # # # # # # # # # # # # # # # # #
 * # # # # # # # # # # # # # # # # # # # #
 * # # # # # # # # # # # # # # # # # # # #
 * # # # # # # # # # # # # # # # # # # # #
 * # # # # # # # # # # # # # # # # # # # #
 * # # # # # # # # # # # # # # # # # # # #
 * # # # # # # # # # # # # # # # # # # # #
 * # # # # # # # # # # # # # # # # # # # #
 * # # # # # # # # # # # # # # # # # # # #
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 * # # # # # # # # # # # # # # # # # # # # * * * * * * * * * * * * * * * * * * * *
 *                                         * * * * * * * * * * * * * * * * * * * *
 *                                         * * * * * * * * * * * * * * * * * * * *
 *                                         * * * * * * * * * * * * * * * * * * * *
 *                                         * * * * * * * * * * * * * * * * * * * *
 *                                         * * * * * * * * * * * * * * * * * * * *
 *                                         * * * * * * * * * * * * * * * * * * * *
 *                                         * * * * * * * * * * * * * * * * * * * *
 *                                         * * * * * * * * * * * * * * * * * * * *
 *                                         * * * * * * * * * * * * * * * * * * * *
 *                                         * * * * * * * * * * * * * * * * * * * *
 */
void Domain::readFromFile(char* path) {
	ifstream in;
	in.open(path, ios::binary);

	readFileStat(in);
	readTimeSetting(in);
	readSaveInterval(in);
	readGridSteps(in);
	readCellAndHaloSize(in);

	readBlockCount(in);

	mBlocks = new Block* [blockCount];

	for (int i = 0; i < blockCount; ++i)
		mBlocks[i] = readBlock(in);

	/*
	 * Чтение количества соединений.

	in >> connectionCount;


	 * Создаем массив указателей на соединения.

	mInterconnects = new Interconnect* [connectionCount];


	 * Читаем соединения.
	 * Функция чтения соединения вовращает указатель на соединение, его записываем в нужную позицию в массиве.

	for (int i = 0; i < connectionCount; ++i)
		mInterconnects[i] = readConnection(in);


	 * Для каждого блока выполняем операцию переноса данных из вектора в масиив.
	 *
	 * Информация о частях границ для отправки/получения переносится из векторов в массивы.
	 * Еак как на момент создания блока неизвестно сколько именно у него "соседей",
	 * сколько частей границ нужно отправлять/получать, изначально данные пишутся в ектор, а затем переносятся в массивы.

	for (int i = 0; i < blockCount; ++i)
		mBlocks[i]->moveTempBorderVectorToBorderArray();*/
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
	in.read((char*)&stepTime, SIZE_DOUBLE);

	cout << "start time:    " << startTime << endl;
	cout << "stop time:     " << stopTime << endl;
	cout << "step time:     " << stepTime << endl;
}

void Domain::readSaveInterval(ifstream& in) {
	in.read((char*)&saveInterval, SIZE_DOUBLE);

	cout << "save interval: " << saveInterval << endl;
}

void Domain::readGridSteps(ifstream& in) {
	in.read((char*)&dx, SIZE_DOUBLE);
	in.read((char*)&dy, SIZE_DOUBLE);
	in.read((char*)&dz, SIZE_DOUBLE);

	cout << "dx:            " << dx << endl;
	cout << "dy:            " << dy << endl;
	cout << "dz:            " << dz << endl;
}

void Domain::readCellAndHaloSize(ifstream& in) {
	in.read((char*)&cellSize, SIZE_INT);
	in.read((char*)&haloSize, SIZE_INT);

	cout << "cell size:     " << cellSize << endl;
	cout << "halo size:     " << haloSize << endl;
}

void Domain::readBlockCount(ifstream& in) {
	in.read((char*)&blockCount, SIZE_INT);

	cout << "block count:   " << blockCount << endl;
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
Block* Domain::readBlock(ifstream& in) {
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
	cout << "Block #" << "<NONE>" << endl;
	cout << "	dimension:     " << dimension << endl;
	cout << "	node:          " << node << endl;
	cout << "	device type:   " << deviceType << endl;
	cout << "	device number: " << deviceNumber << endl;

	for (int j = 0; j < dimension; ++j) {
		in.read((char*)&offset[j], SIZE_INT);
		cout << "	offset" << j << ":           " << count[j] << endl;
	}

	for (int j = 0; j < dimension; ++j) {
		in.read((char*)&count[j], SIZE_INT);
		cout << "	count" << j << ":            " << count[j] << endl;
		total *= count[j];
	}

	for (int j = 0; j < total; ++j) {
		unsigned short int value;
		in.read((char*)&value, SIZE_UN_SH_INT);
		cout << value << " ";
	}

	cout << endl;

	cout << endl << "DON'T CREATE GPU BLOCK! SEE DOMAIN.H includes!!!" << endl;

	if(node == world_rank)
		switch (deviceType) {
			case 0:
				return new BlockCpu(dimension, count[0], count[1], count[2], offset[0], offset[1], offset[2], node, deviceNumber, haloSize, cellSize);
			case 1:
				//return new BlockGpu(length, width, lengthMove, widthMove, world_rank_creator, 0);
			case 2:
				//return new BlockGpu(length, width, lengthMove, widthMove, world_rank_creator, 1);
			case 3:
				//return new BlockGpu(length, width, lengthMove, widthMove, world_rank_creator, 2);
			default:
				return new BlockNull(dimension, count[0], count[1], count[2], offset[0], offset[1], offset[2], node, deviceNumber, haloSize, cellSize);
		}
	else
		return new BlockNull(dimension, count[0], count[1], count[2], offset[0], offset[1], offset[2], node, deviceNumber, haloSize, cellSize);
}

/*
 * Чтение соединения.
 */
Interconnect* Domain::readConnection(ifstream& in) {
	/*
	 * Соединение формируется из
	 * блока источника
	 * блока получателя
	 * стороны (границы) блока получателя
	 * сдвига по границе источника
	 * сдвига по границе получателя
	 */
	int source;
	int destination;

	char borderSide;

	int connectionSourceMove;
	int connectionDestinationMove;
	int borderLength;

	/*
	 * Чтение номера блока-источника и номера блока-получателя
	 */
	in >> source;
	in >> destination;

	/*
	 * Чтение стороны соединения блока-получателя
	 */
	in >> borderSide;

	/*
	 * Сдвиги по границе для блока-источника и блока-получателя.
	 */
	in >> connectionSourceMove;
	in >> connectionDestinationMove;

	/*
	 * Длина границы.
	 */
	in >> borderLength;

	/*
	 * Переменная, которая реально хранит торону.
	 * Целое число, а не символ.
	 */
	int side;

	/*
	 * По счиатнным данным извлекается информация о блоках
	 * Номера потокав, к которым они РЕАЛЬНО приписаны
	 */
	int sourceNode = mBlocks[source]->getNodeNumber();
	int destinationNode = mBlocks[destination]->getNodeNumber();

	/*
	 * Получение типов блоков.
	 * CPU / DEVICE ...
	 */
	int sourceType = mBlocks[source]->getBlockType();
	int destionationType = mBlocks[destination]->getBlockType();

	/*
	 * По считанной букве определяется сторона блока НАЗНАЧЕНИЯ
	 */
	switch (borderSide) {
		case 't':
			side = TOP;
			break;

		case 'l':
			side = LEFT;
			break;

		case 'b':
			side = BOTTOM;
			break;

		case 'r':
			side = RIGHT;
			break;

		default:
			return NULL;
	}

	/*
	 * Функция oppositeBorder возвращает противоположную сторону.
	 * Если граница блока назначения правая - значит у блока источника левая и так далее.
	 *
	 * Блок-источник добавляет себе новую часть границы, которую будет необходимо переслать.
	 * Возвращает указатель на область, в котрую будет помещены данные.
	 * В зависимости от номера потока блока-назаначения и его типа память должна выделяться по-разному.
	 *
	 * Блок-назначения добавляет себе новую область, откуда можно будет достать информацию от блока источника.
	 * Если блоки расположены на одном узле, то область может не выделится.
	 * Блок будет ссылать прямо на область памяти, в которую блок-источник будет помещать данные.
	 */
	double* sourceData = mBlocks[source]->addNewBlockBorder(mBlocks[destination], oppositeBorder(side), connectionSourceMove, borderLength);
	double* destinationData = mBlocks[destination]->addNewExternalBorder(mBlocks[source], side, connectionDestinationMove, borderLength, sourceData);

	/*
	 * Формируется соединение.
	 * оно же и вовращается.
	 */
	return new Interconnect(sourceNode, destinationNode, sourceType, destionationType, borderLength, sourceData, destinationData);
}

/*
 * Получение общего количества узлов сетки.
 * Сумма со всех блоков.
 */
int Domain::getCountGridNodes() {
	int count = 0;
	for (int i = 0; i < blockCount; ++i)
		count += mBlocks[i]->getGridNodeCount();

	return count;
}

/*
 * Заного вычисляется количество повторений для вычислений.
 * Функция носит исключетельно статистический смысл (на данный момент).
 */
int Domain::getRepeatCount() {
	return repeatCount;
}

/*
 * Количество блоков, имеющих тип "центальный процессор".
 */
int Domain::getCpuBlocksCount() {
	int count = 0;
	for (int i = 0; i < blockCount; ++i)
		if( isCPU(mBlocks[i]->getBlockType()) )
			count++;

	return count;
}

/*
 * Количество блоков, имеющих тип "видеокарта".
 */
int Domain::getGpuBlocksCount() {
	int count = 0;
	for (int i = 0; i < blockCount; ++i)
		if(isGPU(mBlocks[i]->getBlockType()))
			count++;

	return count;
}

/*
 * Количество реальных блоков на этом потоке.
 */
int Domain::realBlockCount() {
	int count = 0;
	for (int i = 0; i < blockCount; ++i)
		if( mBlocks[i]->isRealBlock() )
			count++;

	return count;
}

void Domain::saveStateToFile(char* path) {
	cout << endl << "SAVE STATE DON'T WORK" << endl;
	return;
	/*double** resultAll = collectDataFromNode();

	if( world_rank == 0 ) {
		ofstream out;
		out.open(path, ios::binary);

		int save_file_code = SAVE_FILE_CODE;
		int version_major = VERSION_MAJOR;
		int version_minor = VERSION_MINOR;

		out.write((char*)&save_file_code, SIZE_INT);
		out.write((char*)&version_major, SIZE_INT);
		out.write((char*)&version_minor, SIZE_INT);

		out.write((char*)&currentTime, SIZE_DOUBLE);
		out.write((char*)&blockCount, SIZE_INT);

		//out << SAVE_FILE_CODE << endl;
		//out << VERSION_MAJOR << endl;
		//out << VERSION_MINOR << endl;
		//out << currentTime << endl;
		//out << blockCount << endl;

		int length;
		int width;

		double value;

		for (int i = 0; i < blockCount; ++i) {
			length = mBlocks[i]->getLength();
			width = mBlocks[i]->getWidth();

			cout << length << "l " << width << "w ";

			out.write((char*)&length, SIZE_INT);
			out.write((char*)&width, SIZE_INT);

			//out << mBlocks[i]->getLength() << " " << mBlocks[i]->getWidth() << endl;
			for (int j = 0; j < mBlocks[i]->getLength() * mBlocks[i]->getWidth(); ++j) {
				value = resultAll[i][j];
				//out << resultAll[i][j] << " ";
				out.write((char*)&(resultAll[i][j]), SIZE_DOUBLE);
			}
			out << endl;
		}

		out.close();
	}

	if( resultAll != NULL ) {
		for (int i = 0; i < blockCount; ++i)
			delete resultAll[i];
		delete resultAll;
	}*/
}

void Domain::loadStateFromFile(char* blockLocation, char* dataFile) {
	cout << endl << "LOAD STATE DON'T WORK" << endl;
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
	cout << endl << "PRINT STATISTIC INFO DON'T WORK" << endl;
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
