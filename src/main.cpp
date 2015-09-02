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

	int jobId = atoi(argv[1]);
	char* inputFile = argv[2];
	int flags = atoi(argv[3]);
	double stopTime = atof(argv[4]);
	char* saveFile = argv[5];

	/*
	 * Создание основного управляющего класса.
	 * Номер потока, количество потоков и путь к файлу с данными.
	 */
	printf ("DEBUG creating domain...\n ");
	Domain* d = new Domain(world_rank, world_size, inputFile, jobId);
	d->checkOptions(flags, stopTime, saveFile);

	//d->printBlocksToConsole();

	/*
	 * Вычисления.
	 */
	//omp_set_num_threads(1);

	d->saveState(inputFile);

	printf("\n\nBEFORE COMPUTE\n");
	printf ("Running computations %d \n", world_rank);
	time1 = MPI_Wtime();
	d->compute(inputFile);
	time2 = MPI_Wtime();
	printf("\n\nAFTER COMPUTE\n");

	d->saveState(inputFile);

	//d->printBlocksToConsole();

	/*
	 * Вывод информации о времени работы осуществляет только поток с номером 0.
	 * Время работы -  разница между двумя отсечками, котрые были сделаны ранее.
	 */
	d->printStatisticsInfo(inputFile, NULL, time2 - time1, NULL);

	if(d->isNan())
		printf("\n\n\n\nNAN!!!\n\n\n\n");

	delete d;

	/*
	 * Завершение MPI
	 */
	MPI_Finalize();
}
