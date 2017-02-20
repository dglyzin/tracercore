#include "domain/domain.h"

#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>


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

	//domain->printBlocksToConsole();

	/*
	 * Вычисления.
	 */
	domain->saveStateForDraw(inputFile);

	/* Test topology */
/*	printf("MPI world size: %d\n", world_size);
    pid_t pid = getpid();

pid_t tid = syscall(SYS_gettid);

printf("ProcessId=%d, main thread id=%d\n", pid, tid);
#pragma omp parallel
{
#pragma omp single
	{

for (int i = 0; i < 3; ++i) {
#pragma omp task
	{
	pid_t tid2 = syscall(SYS_gettid);
	printf("OMP inside a task with tnum %d and  tid %d \n", omp_get_thread_num(), tid2 );
	int arr[1000000];
	for (int j=0; j<1000000; j++)
		arr[j] = i+j;
	printf("arr[2]=%d\n", arr[2]);
	}
}

	}
}
#pragma omp parallel
{
#pragma omp single
		printf("OMP number of threads %d \n", omp_get_num_threads());

	pid_t tid = syscall(SYS_gettid);
	printf("OMP thread %d with tid %d \n", omp_get_thread_num(), tid );

}

*/

	printf("Running computations mpi rank %d \n", world_rank);

	MPI_Barrier(MPI_COMM_WORLD);
	time1 = omp_get_wtime();//MPI_Wtime();

	domain->compute(inputFile);

	MPI_Barrier(MPI_COMM_WORLD);
	time2 = omp_get_wtime(); //MPI_Wtime();

	domain->saveStateForDraw(inputFile);
	domain->saveStateForLoad(inputFile);

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
