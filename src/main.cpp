#include <mpi.h>
#include <stdlib.h>
#include <cmath>

#include "domain.h"

using namespace std;

int main(int argc, char * argv[]) {
	/*
	 * Инициализация MPI
	 */
	MPI_Init(NULL, NULL);

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

	char* inputFile = argv[1];
	char* outputFile = argv[2];
	int flags = atoi(argv[3]);

	int stepCount = atoi(argv[4]);
	double stopTime = atof(argv[5]);
	char* saveFile = argv[6];
	char* loadFile = argv[7];
	char* statisticsFile = argv[8];

	/*
	 * Создание основного управляющего класса.
	 * Номер потока, количество потоков и путь к файлу с данными.
	 */

	Domain* d = new Domain(world_rank, world_size, inputFile, flags, stepCount, stopTime, loadFile);

	/*
	 * Вычисления.
	 */
	// Получить текущее время
	time1 = MPI_Wtime();
	d->compute(saveFile);
	// Получить текущее время
	time2 = MPI_Wtime();

	/*
	 * Вывод информации о времени работы осуществляет только поток с номером 0.
	 * Время работы -  разница между двумя отсечками, котрые были сделаны ранее.
	 */
	d->printStatisticsInfo(inputFile, outputFile, time2 - time1, statisticsFile);

	/*
	 * Сбор и вывод результата.
	 * Передается второй аргумент командной строки - путь к файлу, в который будет записан результат.
	 */
	d->print(outputFile);

	//delete d;

	/*
	 * Завершение MPI
	 */
	MPI_Finalize();
}
