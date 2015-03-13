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
	/*
	 * Считаем поток с номером 0 (так как он всегда есть) главным.
	 * Он будет собирать со всех результаты вычислений и формировать из них ответ.
	 */
	if(world_rank == 0) {
		/*
		 * Создается матрица по размерам области вычислений
		 * Сюда должны поместиться все блоки.
		 * Это должно быть гарантированно.
		 */
		double** resultAll = new double* [lengthArea];
		for (int i = 0; i < lengthArea; ++i)
			resultAll[i] = new double[widthArea];

		/*
		 * Инициализируется 0. В дальнейшем части области не занятые блоками будут иметь значение 0.
		 */
		for (int i = 0; i < lengthArea; ++i)
			for (int j = 0; j < widthArea; ++j)
				resultAll[i][j] = 0;

		/*
		 * Движемся по массиву блоков и проверяем реальны ли они на данном потоке исполнения.
		 * Если реальны, то пересылка не нужна - это блоки потока 0.
		 * Можно просто забрать данные.
		 *
		 * Если это не реальные блоки, то информация по этим блокам существует на других потоках.
		 * В таком случае ожидается получение информации от потока, который РЕАЛЬНО использовал этот блок.
		 *
		 * В обих случаея используются известные данные блока о его размерах и положениях в области для корректного заполнения результирующей матрицы.
		 */
		for (int i = 0; i < blockCount; ++i) {
			if(mBlocks[i]->isRealBlock()) {
				double* result = mBlocks[i]->getResult();

				for (int j = 0; j < mBlocks[i]->getLength(); ++j)
					for (int k = 0; k < mBlocks[i]->getWidth(); ++k)
						resultAll[j + mBlocks[i]->getLengthMove()][k + mBlocks[i]->getWidthMove()] = result[j * mBlocks[i]->getWidth() + k];
			}
			else
				/*
				 * Получение данных построчно.
				 * Такой способ получения необходим, несмотря на то, что матрица в блоке фактически является одномерным массивом.
				 *
				 * Любой блок может быть смещен в области на какое-то значение по горизонтали и вертикали.
				 * Поэтому принять всю матрицу сразу не возможно.
				 * Каждую строчку нужно отдельно позиционировать в конечной области.
				 */
				for (int j = 0; j < mBlocks[i]->getLength(); ++j)
					MPI_Recv(resultAll[j + mBlocks[i]->getLengthMove()] + mBlocks[i]->getWidthMove(), mBlocks[i]->getWidth(), MPI_DOUBLE, mBlocks[i]->getNodeNumber(), 999, MPI_COMM_WORLD, &status);
		}

		return resultAll;
	}
	else {
		/*
		 * Если это не поток 0, то необходимо переслать данные.
		 * Выполняется это построчно.
		 * (см. описание выше)
		 *
		 * Выполняется проверка на то, что это реальный блок этого потока.
		 */
		for (int i = 0; i < blockCount; ++i) {
			if(mBlocks[i]->isRealBlock()) {
				double* result = mBlocks[i]->getResult();

				for (int j = 0; j < mBlocks[i]->getLength(); ++j)
					MPI_Send(result + (j * mBlocks[i]->getWidth()), mBlocks[i]->getWidth(), MPI_DOUBLE, 0, 999, MPI_COMM_WORLD);
			}
		}

		return NULL;
	}
}

void Domain::count(char* saveFile) {
	/*
	 * Вычисление коэффициентов необходимых для расчета теплопроводности
	 */
	double dX = 1./widthArea;
	double dY = 1./lengthArea;

	/*
	 * Аналогично вышенаписанному
	 */
	double dX2 = dX * dX;
	double dY2 = dY * dY;

	double dT = ( dX2 * dY2 ) / ( 2 * ( dX2 + dY2 ) );

	/*
	 * Выполнение
	 */
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
		saveStateToFile(saveFile);
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
	//computeOneStep(dX2, dY2, dT);
}

void Domain::prepareData() {
#pragma omp task
	{
		for (int i = 0; i < blockCount; ++i)
			if( mBlocks[i]->getBlockType() == GPU && mBlocks[i]->getDeviceNumber() == 0 ) {
				mBlocks[i]->prepareData();
			}
	}

#pragma omp task
	{
		for (int i = 0; i < blockCount; ++i)
			if( mBlocks[i]->getBlockType() == GPU && mBlocks[i]->getDeviceNumber() == 1 ) {
				mBlocks[i]->prepareData();
			}
	}

#pragma omp task
	{
		for (int i = 0; i < blockCount; ++i)
			if( mBlocks[i]->getBlockType() == GPU && mBlocks[i]->getDeviceNumber() == 2 ) {
				mBlocks[i]->prepareData();
			}
	}

#pragma omp task
	{
		for (int i = 0; i < blockCount; ++i)
			if( mBlocks[i]->getBlockType() == CPU ) {
				mBlocks[i]->prepareData();
			}
	}
}

void Domain::computeOneStep(double dX2, double dY2, double dT) {
#pragma omp task
	{
		for (int i = 0; i < blockCount; ++i)
			if( mBlocks[i]->getBlockType() == GPU && mBlocks[i]->getDeviceNumber() == 0 ) {
				mBlocks[i]->computeOneStep(dX2, dY2, dT);
			}
	}

#pragma omp task
	{
		for (int i = 0; i < blockCount; ++i)
			if( mBlocks[i]->getBlockType() == GPU && mBlocks[i]->getDeviceNumber() == 1 ) {
				mBlocks[i]->computeOneStep(dX2, dY2, dT);
			}
	}

#pragma omp task
	{
		for (int i = 0; i < blockCount; ++i)
			if( mBlocks[i]->getBlockType() == GPU && mBlocks[i]->getDeviceNumber() == 2 ) {
				mBlocks[i]->computeOneStep(dX2, dY2, dT);
			}
	}

#pragma omp task
	{
		for (int i = 0; i < blockCount; ++i)
			if( mBlocks[i]->getBlockType() == CPU ) {
				mBlocks[i]->computeOneStep(dX2, dY2, dT);
			}
	}
}

void Domain::computeOneStepBorder(double dX2, double dY2, double dT) {
#pragma omp task
	{
		for (int i = 0; i < blockCount; ++i)
			if( mBlocks[i]->getBlockType() == GPU && mBlocks[i]->getDeviceNumber() == 0 ) {
				mBlocks[i]->computeOneStepBorder(dX2, dY2, dT);
			}
	}

#pragma omp task
	{
		for (int i = 0; i < blockCount; ++i)
			if( mBlocks[i]->getBlockType() == GPU && mBlocks[i]->getDeviceNumber() == 1 ) {
				mBlocks[i]->computeOneStepBorder(dX2, dY2, dT);
			}
	}

#pragma omp task
	{
		for (int i = 0; i < blockCount; ++i)
			if( mBlocks[i]->getBlockType() == GPU && mBlocks[i]->getDeviceNumber() == 2 ) {
				mBlocks[i]->computeOneStepBorder(dX2, dY2, dT);
			}
	}

#pragma omp task
	{
		for (int i = 0; i < blockCount; ++i)
			if( mBlocks[i]->getBlockType() == CPU ) {
				mBlocks[i]->computeOneStepBorder(dX2, dY2, dT);
			}
	}
}

void Domain::computeOneStepCenter(double dX2, double dY2, double dT) {
#pragma omp task
	{
		for (int i = 0; i < blockCount; ++i)
			if( mBlocks[i]->getBlockType() == GPU && mBlocks[i]->getDeviceNumber() == 0 ) {
				mBlocks[i]->computeOneStepCenter(dX2, dY2, dT);
			}
	}

#pragma omp task
	{
		for (int i = 0; i < blockCount; ++i)
			if( mBlocks[i]->getBlockType() == GPU && mBlocks[i]->getDeviceNumber() == 1 ) {
				mBlocks[i]->computeOneStepCenter(dX2, dY2, dT);
			}
	}

#pragma omp task
	{
		for (int i = 0; i < blockCount; ++i)
			if( mBlocks[i]->getBlockType() == GPU && mBlocks[i]->getDeviceNumber() == 2 ) {
				mBlocks[i]->computeOneStepCenter(dX2, dY2, dT);
			}
	}

#pragma omp task
	{
		for (int i = 0; i < blockCount; ++i)
			if( mBlocks[i]->getBlockType() == CPU ) {
				mBlocks[i]->computeOneStepCenter(dX2, dY2, dT);
			}
	}
}

void Domain::swapBlockMatrix() {
#pragma omp task
	{
		for (int i = 0; i < blockCount; ++i)
			if( mBlocks[i]->getBlockType() == GPU && mBlocks[i]->getDeviceNumber() == 0 ) {
				mBlocks[i]->swapMatrix();
			}
	}

#pragma omp task
	{
		for (int i = 0; i < blockCount; ++i)
			if( mBlocks[i]->getBlockType() == GPU && mBlocks[i]->getDeviceNumber() == 1 ) {
				mBlocks[i]->swapMatrix();
			}
	}

#pragma omp task
	{
		for (int i = 0; i < blockCount; ++i)
			if( mBlocks[i]->getBlockType() == GPU && mBlocks[i]->getDeviceNumber() == 2 ) {
				mBlocks[i]->swapMatrix();
			}
	}

#pragma omp task
	{
		for (int i = 0; i < blockCount; ++i)
			if( mBlocks[i]->getBlockType() == CPU ) {
				mBlocks[i]->swapMatrix();
			}
	}
}

void Domain::print(char* path) {
	double** resultAll = collectDataFromNode();

	/*
	 * После формирования результирующей матрицы производится вывод в файл.
	 * Путь к файлу передается параметром этой функции.
	 * Арумент командной строки.
	 */

	if( world_rank == 0 ) {
		ofstream out;
		out.open(path);

		for (int i = 0; i < lengthArea; ++i) {
			for (int j = 0; j < widthArea; ++j)
				out << i << " " << j << " " << resultAll[i][j] << endl;
			out << endl;
		}

		out.close();
	}

	if( resultAll != NULL ) {
		for (int i = 0; i < lengthArea; ++i)
			delete resultAll[i];
		delete resultAll;
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
	in.open(path);

	readTimeSetting(in);

	/*
	 * Чтение размеров области
	 */
	readLengthAndWidthArea(in);

	/*
	 * Чтение количества блоков
	 */
	in >> blockCount;

	/*
	 * Создаем массив указателей на блоки.
	 * Его длина = количество блоков.
	 */
	mBlocks = new Block* [blockCount];

	/*
	 * Чтение блоков.
	 * Функция чтения блоков возвращает указатель на блок, его записываем в нужную позицию в массиве.
	 */
	for (int i = 0; i < blockCount; ++i)
		mBlocks[i] = readBlock(in);

	/*
	 * Чтение количества соединений.
	 */
	in >> connectionCount;

	/*
	 * Создаем массив указателей на соединения.
	 */
	mInterconnects = new Interconnect* [connectionCount];

	/*
	 * Читаем соединения.
	 * Функция чтения соединения вовращает указатель на соединение, его записываем в нужную позицию в массиве.
	 */
	for (int i = 0; i < connectionCount; ++i)
		mInterconnects[i] = readConnection(in);

	/*
	 * Для каждого блока выполняем операцию переноса данных из вектора в масиив.
	 *
	 * Информация о частях границ для отправки/получения переносится из векторов в массивы.
	 * Еак как на момент создания блока неизвестно сколько именно у него "соседей",
	 * сколько частей границ нужно отправлять/получать, изначально данные пишутся в ектор, а затем переносятся в массивы.
	 */
	for (int i = 0; i < blockCount; ++i)
		mBlocks[i]->moveTempBorderVectorToBorderArray();
}

/*
 * Чтение размеров области.
 */
void Domain::readLengthAndWidthArea(ifstream& in) {
	in >> lengthArea;
	in >> widthArea;
}

void Domain::readTimeSetting(ifstream& in) {
	in >> stepTime;
	in >> stopTime;
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
	int length;
	int width;

	int lengthMove;
	int widthMove;

	int world_rank_creator;

	int type = 0;

	/*
	 * Чтение размеров блока.
	 */
	in >> length;
	in >> width;

	/*
	 * Координаты.
	 * Свдиги по длини и ширине
	 */
	in >> lengthMove;
	in >> widthMove;

	/*
	 * Номер потока, который СОЗДАСТ это блок
	 */
	in >> world_rank_creator;

	/*
	 * Чтение типа блока.
	 * 0 - центральный процессор
	 * 1 - видеокарта 1
	 * 2 - видеокарта 2
	 * 3 - видеокарта 3
	 */
	in >> type;

	/*
	 * Если номер потока исполнения и номер предписанного потока совпадают, то будет сформирован реальный блок.
	 * В противном случае блок-заглушка.
	 *
	 * Предписанный поток задается в файле.
	 * Предписанный поток - поток, который должен иметь этот блок в качестве реального блока.
	 *
	 * В зависимости от считанного типа будет создан либо блок для центрального процессора, либо блок для одной их видеокарт.
	 */
	if(world_rank_creator == world_rank)
		switch (type) {
			case 0:
				return new BlockCpu(length, width, lengthMove, widthMove, world_rank_creator, 0);
			case 1:
				return new BlockGpu(length, width, lengthMove, widthMove, world_rank_creator, 0);
			case 2:
				return new BlockGpu(length, width, lengthMove, widthMove, world_rank_creator, 1);
			case 3:
				return new BlockGpu(length, width, lengthMove, widthMove, world_rank_creator, 2);
			default:
				return new BlockNull(length, width, lengthMove, widthMove, world_rank_creator, 0);
		}
	else
		return new BlockNull(length, width, lengthMove, widthMove, world_rank_creator, 0);
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
		count += mBlocks[i]->getCountGridNodes();

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
int Domain::getCountCpuBlocks() {
	int count = 0;
	for (int i = 0; i < blockCount; ++i)
		if( isCPU(mBlocks[i]->getBlockType()) )
			count++;

	return count;
}

/*
 * Количество блоков, имеющих тип "видеокарта".
 */
int Domain::getCountGpuBlocks() {
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
	double** resultAll = collectDataFromNode();

	if( world_rank == 0 ) {
		ofstream out;
		out.open(path);

		out << currentTime << endl;
		out << lengthArea << " " << widthArea << endl;

		for (int i = 0; i < lengthArea; ++i) {
			for (int j = 0; j < widthArea; ++j)
				out << resultAll[i][j] << " ";
			out << endl;
		}

		out.close();
	}

	if( resultAll != NULL ) {
		for (int i = 0; i < lengthArea; ++i)
			delete resultAll[i];
		delete resultAll;
	}
}

void Domain::loadStateFromFile(char* blockLocation, char* dataFile) {
	readFromFile(blockLocation);

	ifstream in;
	in.open(dataFile);

	int length;
	int width;

	in >> currentTime;
	in >> length;
	in >> width;

	if( length != lengthArea || width != widthArea ) {
		cout << endl << "Critical error!" << endl;
		exit(1);
	}

	double** matrix = new double* [lengthArea];
	for (int i = 0; i < lengthArea; ++i)
		matrix[i] = new double [widthArea];

	for (int i = 0; i < lengthArea; ++i)
		for (int j = 0; j < widthArea; ++j)
			in >> matrix[i][j];

	for (int i = 0; i < blockCount; ++i) {
		if( mBlocks[i]->isRealBlock() ) {

			double* data = new double [mBlocks[i]->getLength() * mBlocks[i]->getWidth()];

			for (int j = 0; j < mBlocks[i]->getLength(); ++j)
				for (int k = 0; k < mBlocks[i]->getWidth(); ++k)
					data[j * mBlocks[i]->getWidth() + k] = matrix[	j + mBlocks[i]->getLengthMove()	][	k + mBlocks[i]->getWidthMove()	];

			mBlocks[i]->loadData(data);

			delete data;
		}
	}
}

void Domain::printStatisticsInfo(char* inputFile, char* outputFile, double calcTime, char* statisticsFile) {
	int countGridNodes = getCountGridNodes();
	int repeatCount = getRepeatCount();
	double speed = (double)(countGridNodes) * repeatCount / calcTime / 1000000;

	if ( flags & STATISTICS ) {
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

		/*for (int i = 0; i < world_size; ++i) {
			int cpuCount = 0;
			int gpuCount = 0;

			for (int j = 0; j < blockCount; ++j) {
				if( mBlocks[i]->getNodeNumber()  == i ) {
					if( isCPU( mBlocks[j]->getBlockType() ) )
						cpuCount++;

					if( isGPU( mBlocks[j]->getBlockType() ) )
						gpuCount++;
				}
			}

			out << "Thread #" << i << " CPU blocks: " << cpuCount << " GPU blocks: " << gpuCount << endl << endl;

		}*/

		out << "############################################################" << endl;

		out.close();
	}
	else {
		cout << "############################################################" << endl;
		cout.precision(5);
		cout << endl <<
				"Input file:   " << inputFile << endl <<
				"Output file:  " << outputFile << endl <<
				"Node count:   " << countGridNodes << endl <<
				"Repeat count: " << repeatCount << endl <<
				"Time:         " << calcTime << endl <<
				"Speed (10^6): " << speed << endl <<
				endl;

		/*for (int i = 0; i < world_size; ++i) {
			int cpuCount = 0;
			int gpuCount = 0;

			if( world_rank == 0 ) {
				MPI_Recv(&cpuCount, 1, MPI_INT, i, 999, MPI_COMM_WORLD, &status);
				MPI_Recv(resultAll[j + mBlocks[i]->getLengthMove()] + mBlocks[i]->getWidthMove(), mBlocks[i]->getWidth(), MPI_DOUBLE, mBlocks[i]->getNodeNumber(), 999, MPI_COMM_WORLD, &status);
			}

			cout << "Thread #" << i << " CPU blocks: " << cpuCount << " GPU blocks: " << gpuCount << endl << endl;

		}*/

		cout << "############################################################" << endl;
	}


}
