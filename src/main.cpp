#include "domain.h"

#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include "logger.h"

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
	char* binaryFileName = argv[0];
	char* inputFile = argv[1];
	int flags = atoi(argv[2]);
	double stopTime = atof(argv[3]);
	char* fileForLoad = argv[4];

	/*
	 * Создание основного управляющего класса.
	 * Номер потока, количество потоков и путь к файлу с данными.
	 */
	time_t now = time(0);
	printf("\n");
	printwcts("Welcome to computing core!\n", LL_INFO);
    printwts("Initital timestamp is " + ToString(now) +"\n" , now, LL_INFO );
	printwcts("SLURM JOB " + ToString(getenv("SLURM_JOB_ID")) +" STARTED\n", LL_INFO);

	char processor_name[MPI_MAX_PROCESSOR_NAME];
	int name_len;
	MPI_Get_processor_name(processor_name, &name_len);

	printwcts("Mpi rank " + ToString(world_rank) + " started on node " + ToString(processor_name) + " \n", LL_INFO);


	printwcts("creating domain...\n", LL_DEBUG);
	int jobId = -1;
	//TODO get jobId from command line when core is launched by webUI
	Domain* domain = new Domain(world_rank, world_size, inputFile, binaryFileName, jobId);
	domain->checkOptions(flags, stopTime, fileForLoad);

	//domain->printBlocksToConsole();

	/*
	 * Вычисления.
	 */
	domain->saveStateForDraw(inputFile, domain->getEntirePlotValues() );

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



	MPI_Barrier(domain->getWorkerComm());
	time1 = omp_get_wtime();//MPI_Wtime();

	domain->compute(inputFile);

	MPI_Barrier(domain->getWorkerComm());
	time2 = omp_get_wtime(); //MPI_Wtime();

	//domain->saveStateForDraw(inputFile,domain->getEntirePlotValues());
	//domain->saveStateForLoad(inputFile);
	//domain->saveStateForDrawDenseOutput(inputFile);

	/*
	 * Вывод информации о времени работы осуществляет только поток с номером 0.
	 * Время работы -  разница между двумя отсечками, котрые были сделаны ранее.
	 */
	if (domain->isNan())
		printwcts("\n\n\n\nNAN!!!\n\n\n\n",LL_INFO);

	//domain->printBlocksToConsole();

	domain->printStatisticsInfo(inputFile, NULL, time2 - time1, NULL);

	delete domain;

	/*
	 * Завершение MPI
	 */
	MPI_Finalize();
}
