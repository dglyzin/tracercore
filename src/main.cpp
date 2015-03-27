#include <mpi.h>
#include <stdlib.h>
#include <cmath>

#include "domain.h"

using namespace std;

void testRead(char* path);

int main(int argc, char * argv[]) {
	/*
	 * TODO Границы - пересылка - расчет центра
	 */

	/*
	 * При запуске обязательно следует указывать 2 аргумента.
	 * Пусть к файлу с исходными данными
	 * Путь к файлу, в который будет записан результат.
	 */

	/*
	 * Инициализация MPI
	 */
	MPI_Init(NULL, NULL);

	testRead("brusselator_2block.bin");

	/*
	 * Для каждого потока получаем его номер и общее количество потоков.
	 */
	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	/*
	 * Переменные для расчета время работы.
	 */
	double time1, time2;

	/*
	 * Создание основного управляющего класса.
	 * Номер потока, количество потоков и путь к файлу с данными.
	 */

	char* inputFile = argv[1];
	char* outputFile = argv[2];
	int flags = atoi(argv[3]);

	int stepCount = atoi(argv[4]);
	double stopTime = atof(argv[5]);
	char* saveFile = argv[6];
	char* loadFile = argv[7];
	char* statisticsFile = argv[8];

	//Domain* d = new Domain(world_rank, world_size, inputFile, flags, stepCount, stopTime, loadFile);

	/*
	 * Вычисления.
	 */
	// Получить текущее время
	time1 = MPI_Wtime();
	//d->compute(saveFile);
	// Получить текущее время
	time2 = MPI_Wtime();

	/*
	 * Вывод информации о времени работы осуществляет только поток с номером 0.
	 * Время работы -  разница между двумя отсечками, котрые были сделаны ранее.
	 */
	//d->printStatisticsInfo(inputFile, outputFile, time2 - time1, statisticsFile);

	/*
	 * Сбор и вывод результата.
	 * Передается второй аргумент командной строки - путь к файлу, в который будет записан результат.
	 */
	//d->print(outputFile);

	//delete d;

	/*
	 * Завершение MPI
	 */
	MPI_Finalize();
}

void testRead(char* path) {
	ifstream in;
	in.open(path, ios::binary);

	char fileType;
	char versionMajor;
	char versionMinor;

	double startTime;
	double stopTime;
	double stepTime;

	double saveInterval;

	double dx;
	double dy;
	double dz;

	int cellSize;
	int haloSize;

	int blockCount;

	in.read((char*)&fileType, 1);
	in.read((char*)&versionMajor, 1);
	in.read((char*)&versionMinor, 1);

	in.read((char*)&startTime, 8);
	in.read((char*)&stopTime, 8);
	in.read((char*)&stepTime, 8);

	in.read((char*)&saveInterval, 8);

	in.read((char*)&dx, 8);
	in.read((char*)&dy, 8);
	in.read((char*)&dz, 8);

	in.read((char*)&cellSize, 4);
	in.read((char*)&haloSize, 4);

	in.read((char*)&blockCount, 4);

	cout << endl;
	cout << "file type:     " << (unsigned int)fileType << endl;
	cout << "version major: " << (unsigned int)versionMajor << endl;
	cout << "version minor: " << (unsigned int)versionMinor << endl;
	cout << "start time:    " << startTime << endl;
	cout << "stop time:     " << stopTime << endl;
	cout << "step time:     " << stepTime << endl;
	cout << "save interval: " << saveInterval << endl;
	cout << "dx:            " << dx << endl;
	cout << "dy:            " << dy << endl;
	cout << "dz:            " << dz << endl;
	cout << "cell size:     " << cellSize << endl;
	cout << "halo size:     " << haloSize << endl;
	cout << "block count:   " << blockCount << endl;

	for (int i = 0; i < blockCount; ++i) {
		int dimension;
		int node;
		int deviceType;
		int deviceNumber;

		int total = 1;


		in.read((char*)&dimension, 4);
		in.read((char*)&node, 4);
		in.read((char*)&deviceType, 4);
		in.read((char*)&deviceNumber, 4);

		cout << endl;
		cout << "Block #" << i << endl;
		cout << "	dimension:     " << dimension << endl;
		cout << "	node:          " << node << endl;
		cout << "	device type:   " << deviceType << endl;
		cout << "	device number: " << deviceNumber << endl;

		for (int j = 0; j < dimension; ++j) {
			int count;
			in.read((char*)&count, 4);
			cout << "	#" << j << ":            " << count << endl;
		}

		for (int j = 0; j < dimension; ++j) {
			int count;
			in.read((char*)&count, 4);
			cout << "	*" << j << ":            " << count << endl;
			total *= count;
		}

		for (int j = 0; j < total; ++j) {
			unsigned short int value;
			in.read((char*)&value, 2);
			cout << value << " ";
		}

		cout << endl;
	}

	int connectionCount;
	in.read((char*)&connectionCount, 4);

	cout << "connection:    " << connectionCount << endl;

	for (int i = 0; i < connectionCount; ++i) {
		int dimension;
		in.read((char*)&dimension, 4);

		cout << "Interconnect #" << i << endl;
		for (int j = 0; j < dimension; ++j) {
			int length;
			in.read((char*)&length, 4);
			cout << "	#" << j << ":            " << length << endl;
		}

		int sourceBlock;
		int destinationBlock;

		int sourceSide;
		int destinationSide;

		in.read((char*)&sourceBlock, 4);
		in.read((char*)&destinationBlock, 4);
		in.read((char*)&sourceSide, 4);
		in.read((char*)&destinationSide, 4);

		cout << "	source block:      " << sourceBlock << endl;
		cout << "	destination block: " << destinationBlock << endl;
		cout << "	sourse side:       " << sourceSide << endl;
		cout << "	destination side:  " << destinationSide << endl;

		for (int j = 0; j < dimension; ++j) {
			int count;
			in.read((char*)&count, 4);
			cout << "	#" << j << ":            " << count << endl;
		}

		for (int j = 0; j < dimension; ++j) {
			int count;
			in.read((char*)&count, 4);
			cout << "	*" << j << ":            " << count << endl;
		}
	}

	in.close();
}
