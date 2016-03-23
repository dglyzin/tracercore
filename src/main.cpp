#include "domain.h"

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

	//int jobId = atoi(argv[1]);
	char* inputFile = argv[1];
	int flags = atoi(argv[2]);
	double stopTime = atof(argv[3]);
	char* saveFile = argv[4];

	/*
	 * Создание основного управляющего класса.
	 * Номер потока, количество потоков и путь к файлу с данными.
	 */
	printf("SLURM JOB %s STARTED\n ", getenv("SLURM_JOB_ID"));
	printf("DEBUG creating domain...\n ");
	Domain* domain = new Domain(world_rank, world_size, inputFile);
	domain->checkOptions(flags, stopTime, saveFile);

	/*
	 * Вычисления.
	 */
	domain->saveState(inputFile);

	printf("Running computations %d \n", world_rank);
	time1 = MPI_Wtime();
	domain->compute(inputFile);
	time2 = MPI_Wtime();

	domain->saveState(inputFile);

	/*
	 * Вывод информации о времени работы осуществляет только поток с номером 0.
	 * Время работы -  разница между двумя отсечками, котрые были сделаны ранее.
	 */
	if (domain->isNan())
		printf("\n\n\n\nNAN!!!\n\n\n\n");

	//domain->printBlocksToConsole();

	domain->printStatisticsInfo(inputFile, NULL, time2 - time1, NULL);

	delete domain;

	/*
	 * Завершение MPI
	 */
	MPI_Finalize();
}
