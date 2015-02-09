/*
 * Domain.cpp
 *
 *  Created on: 22 янв. 2015 г.
 *      Author: frolov
 */

#include "domain.h"

using namespace std;

Domain::Domain(int _world_rank, int _world_size, char* path) {
	world_rank = _world_rank;
	world_size = _world_size;

	readFromFile(path);
}

Domain::~Domain() {
	// TODO Auto-generated destructor stub
}

void Domain::count() {
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
	 * Вычисление количества необходимых итераций
	 */
	int repeatCount = (int)(1 / dT) + 1;
	//repeatCount = 1000;
	//printf("\nREPEAT COUNT NOT RIGHT!!!\n");

	/*
	 * Выполнение
	 */
	for (int i = 0; i < repeatCount; ++i)
		nextStep(dX2, dY2, dT);
}

void Domain::nextStep(double dX2, double dY2, double dT) {
	/*
	 * Все блоки подготавливают данные для пересылки
	 */
	for (int i = 0; i < blockCount; ++i)
		mBlocks[i]->prepareData();

	/*
	 * Все данные пересылаются
	 */
	for (int i = 0; i < connectionCount; ++i)
		mInterconnects[i]->sendRecv(world_rank);

	/*
	 * Перерасчет данных
	 */
	omp_set_num_threads(realBlockCount());
#pragma omp parallel
	{
#pragma omp for
		for (int i = 0; i < blockCount; ++i)
			mBlocks[i]->courted(dX2, dY2, dT);
	}
}

void Domain::print(char* path) {
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
		double** resaultAll = new double* [lengthArea];
		for (int i = 0; i < lengthArea; ++i)
			resaultAll[i] = new double[widthArea];

		/*
		 * Инициализируется 0. В дальнейшем части области не занятые блоками будут иметь значение 0.
		 */
		for (int i = 0; i < lengthArea; ++i)
			for (int j = 0; j < widthArea; ++j)
				resaultAll[i][j] = 0;

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
				double* resault = mBlocks[i]->getResault();

				for (int j = 0; j < mBlocks[i]->getLength(); ++j)
					for (int k = 0; k < mBlocks[i]->getWidth(); ++k)
						resaultAll[j + mBlocks[i]->getLenghtMove()][k + mBlocks[i]->getWidthMove()] = resault[j * mBlocks[i]->getWidth() + k];
			}
			else
				/*
				 * Данные получаем все сразу
				 */
				for (int j = 0; j < mBlocks[i]->getLength(); ++j)
					MPI_Recv(resaultAll[j + mBlocks[i]->getLenghtMove()] + mBlocks[i]->getWidthMove(), mBlocks[i]->getWidth(), MPI_DOUBLE, mBlocks[i]->getNodeNumber(), 999, MPI_COMM_WORLD, &status);
		}

		/*
		 * После формирования результирующей матрицы производится вывод в файл.
		 * Путь к файлу передается параметром этой функции.
		 * Арумент командной строки.
		 */
		FILE* out = fopen(path, "wb");

		for (int i = 0; i < lengthArea; ++i) {
			for (int j = 0; j < widthArea; ++j)
				fprintf(out, "%d %d %f\n", i, j, resaultAll[i][j]);
			fprintf(out, "\n");
		}

		fclose(out);
	}
	else {
		/*
		 * Если это не поток 0, то необходимо переслать данные.
		 * Выполняется это построчно
		 *
		 * Выполняется проверка на то, что это реальный блок этого потока.
		 */
		for (int i = 0; i < blockCount; ++i) {
			if(mBlocks[i]->isRealBlock()) {
				double* resault = mBlocks[i]->getResault();

				for (int j = 0; j < mBlocks[i]->getLength(); ++j)
					MPI_Send(resault + (j * mBlocks[i]->getWidth()), mBlocks[i]->getWidth(), MPI_DOUBLE, 0, 999, MPI_COMM_WORLD);
			}
		}
	}
}

void Domain::readFromFile(char* path) {
	ifstream in;
	in.open(path);

	readLengthAndWidthArea(in);
	in >> blockCount;

	mBlocks = new Block* [blockCount];

	for (int i = 0; i < blockCount; ++i)
		mBlocks[i] = readBlock(in);

	in >> connectionCount;

	mInterconnects = new Interconnect* [connectionCount];

	for (int i = 0; i < connectionCount; ++i)
		mInterconnects[i] = readConnection(in);
}

void Domain::readLengthAndWidthArea(ifstream& in) {
	in >> lengthArea;
	in >> widthArea;
}

Block* Domain::readBlock(ifstream& in) {
	int length;
	int width;

	int lengthMove;
	int widthMove;

	int world_rank_creator;

	int type = 0;

	in >> length;
	in >> width;

	in >> lengthMove;
	in >> widthMove;

	in >> world_rank_creator;

	in >> type;

	/*
	 * Если номер потока исполнения и номер предписанного потока совпадают, то будет сформирован реальный блок.
	 * В противном случае блок-заглушка.
	 *
	 * Предписанный поток задается в файле.
	 * Предписанный поток - поток, который должен иметь этот блок в качестве реального блока.
	 */
	if(world_rank_creator == world_rank)
		switch (type) {
			case 0:
				return new BlockCpu(length, width, lengthMove, widthMove, world_rank_creator);
			case 1:
				return new BlockGpu(length, width, lengthMove, widthMove, world_rank_creator, getDeviceNumber(DEVICE0));
			case 2:
				return new BlockGpu(length, width, lengthMove, widthMove, world_rank_creator, getDeviceNumber(DEVICE1));
			case 3:
				return new BlockGpu(length, width, lengthMove, widthMove, world_rank_creator, getDeviceNumber(DEVICE2));
			default:
				return new BlockNull(length, width, lengthMove, widthMove, world_rank_creator);
		}
	else
		return new BlockNull(length, width, lengthMove, widthMove, world_rank_creator);
}

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
	 * Чтение данных из файла
	 */
	in >> source;
	in >> destination;

	in >> borderSide;

	in >> connectionSourceMove;
	in >> connectionDestinationMove;

	in >> borderLength;

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
			// TODO Рассматривать случай?
			return NULL;
	}

	/*
	 * Получаются указателя на границы, уже со сдвигом
	 * Функция oppositeBorder возвращает противоположную сторону.
	 * Если граница блока назначения правая - значит у блока источника левая и так далее.
	 */


	double* sourceData = mBlocks[source]->getBorderBlockData( oppositeBorder(side), connectionSourceMove );
	double* destinationData = mBlocks[destination]->getExternalBorderData(side, connectionDestinationMove);


	/*
	 * Если блок назначения реален для данного потока,то тип границы должен быть изменен, чтобы вчисления были корректны.
	 * тип границы, сторона, сдвиг и длина границы
	 */
	if(mBlocks[destination]->isRealBlock())
		mBlocks[destination]->setPartBorder(BY_ANOTHER_BLOCK, side, connectionDestinationMove, borderLength);

	/*
	 * Формируется соединение.
	 * оно же и вовращается.
	 */
	return new Interconnect(sourceNode, destinationNode, sourceType, destionationType, borderLength, sourceData, destinationData);
}

int Domain::getCountGridNodes() {
	int count = 0;
	for (int i = 0; i < blockCount; ++i)
		count += mBlocks[i]->getCountGridNodes();

	return count;
}

int Domain::getRepeatCount() {
	double dX = 1./widthArea;
	double dY = 1./lengthArea;

	double dX2 = dX * dX;
	double dY2 = dY * dY;

	double dT = ( dX2 * dY2 ) / ( 2 * ( dX2 + dY2 ) );

	return (int)(1 / dT) + 1;
}

int Domain::getCountCpuBlocks() {
	int count = 0;
	for (int i = 0; i < blockCount; ++i)
		if(mBlocks[i]->getBlockType() == CPU )
			count++;

	return count;
}

int Domain::getCountGpuBlocks() {
	int count = 0;
	for (int i = 0; i < blockCount; ++i)
		if(mBlocks[i]->getBlockType() != CPU && mBlocks[i]->getBlockType() != NULL_BLOCK)
			count++;

	return count;
}

int Domain::realBlockCount() {
	int count = 0;
	for (int i = 0; i < blockCount; ++i)
		if( mBlocks[i]->isRealBlock() )
			count++;

	return count;
}
