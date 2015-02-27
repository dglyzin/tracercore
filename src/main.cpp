#include <mpi.h>
#include <stdlib.h>
#include <cmath>

#include "domain.h"

int main(int argc, char * argv[]) {
	/*
	 * При запуске обязательно следует указывать 2 аргумента.
	 * Пусть к файлу с исходными данными
	 * Путь к файлу, в который будет записан результат.
	 */

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

	/*
	 * Создание основного управляющего класса.
	 * Номер потока, количество потоков и путь к файлу с данными.
	 */

	char* inputFile = argv[1];
	char* outputFile = argv[2];
	int flags = atoi(argv[3]);

	double percentageCompletion = atoi(argv[4]);
	char* saveFile = argv[5];
	char* loadFile = argv[6];


	Domain* d = new Domain(world_rank, world_size, inputFile, flags, percentageCompletion, loadFile);

	if(world_rank == 0) {
		printf("\n\n");
		printf("File:         %s\n", argv[1]);
		printf("Node count:   %d\n", d->getCountGridNodes());
		printf("Repeat count: %d\n", d->getRepeatCount());
	}

	/*
	 * Вычисления.
	 */
	// Получить текущее время
	time1 = MPI_Wtime();
	d->count(saveFile);
	// Получить текущее время
	time2 = MPI_Wtime();

	/*
	 * Вывод информации о времени работы осуществляет только поток с номером 0.
	 * Время работы -  разница между двумя отсечками, котрые были сделаны ранее.
	 */
	if(world_rank == 0) {
		double calcTime = time2 - time1;
		printf("Time:         %f\n", calcTime);
		printf("Speed:        %f\n", (double)(d->getCountGridNodes()) * d->getRepeatCount() / calcTime);
		printf("\n");
	}

	printf("\nThread #%d CPU blocks: %d, GPU blocks: %d\n", world_rank, d->getCountCpuBlocks(), d->getCountGpuBlocks());

	/*
	 * Сбор и вывод результата.
	 * Передается второй аргумент командной строки - путь к файлу, в который будет записан результат.
	 */
	d->print(outputFile);
	//printf("\nNOT PRINT TO FILE!!!\n");

	delete d;

	/*
	 * Завершение MPI
	 */
	MPI_Finalize();
}
